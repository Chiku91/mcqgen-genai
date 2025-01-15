import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,create_and_print_dataframe
from src.mcqgenerator.logger import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()

key=os.getenv("OPENAI_API_KEY")

llm= ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo", temperature=0.7)

TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, create exactly {number} multiple choice questions for {subject} students in a {tone} tone. 
Each question should be unique and related to the text. Ensure that all questions are clear, unambiguous, and well-formed, ensuring that each question is of good quality.
The format for the response should strictly follow this structure:
{{
  "quiz": [
    {{
      "mcq": "Your question here",
      "options": {{
        "a": "Choice A",
        "b": "Choice B",
        "c": "Choice C",
        "d": "Choice D"
      }},
      "answer": "correct answer"
    }} 
  ]
}}
Ensure that there are {number} MCQs and no more than that. If less than {number} MCQs are generated, continue generating until the exact number is produced.
"""
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone"],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt)

TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students, evaluate the complexity of the questions and provide a complete analysis of the quiz.
If the quiz is not at par with the cognitive and analytical abilities of the students, update the quiz questions that need to be changed and adjust the tone so it perfectly fits the student abilities.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],  # Only subject and quiz are needed for review chain
    template=TEMPLATE2
)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt)

# User-defined inputs
NUMBER = 5  # Number of MCQs the user wants
SUBJECT = "Programming"
TONE = "friendly"

# Define the callback function to log the result
def openai_callback(result, *args, **kwargs):
    print("OpenAI Callback triggered!")
    print(f"Result: {result}")

# Use get_openai_callback directly and capture the output
with get_openai_callback() as callback:
    # Step 1: Generate the quiz using the first chain
    input_data = {
        "text": TEXT,  # Read text from file dynamically
        "number": NUMBER,
        "subject": SUBJECT,
        "tone": TONE
    }

    # Run the quiz generation chain
    quiz_output = ""
    while True:
        try:
            # Generate the quiz
            quiz_output = quiz_chain.run(input_data)  # Run the chain to generate the quiz

            # Log the raw output from the quiz generation chain for inspection
            print("Raw response from quiz generation chain:")
            print(quiz_output)

            # Check if the output is empty
            if not quiz_output.strip():
                print("Error: Received empty response from the quiz generation model.")
                quiz_output_dict = {}
            else:
                # Try parsing the response as JSON
                try:
                    # Attempt to load the response as JSON (this is now more strict)
                    quiz_output_dict = json.loads(quiz_output)  # Parse the output as JSON

                    # Check if the generated quiz contains the correct number of MCQs
                    quiz_list = quiz_output_dict.get("quiz", [])
                    if len(quiz_list) >= NUMBER:
                        break  # Exit the loop if we have enough questions
                    else:
                        print(f"Generated {len(quiz_list)} MCQs, but {NUMBER} are required. Re-generating...")
                except json.JSONDecodeError as e:
                    print(f"Error parsing quiz output as JSON: {e}")
                    print(f"Raw output received: {quiz_output}")  # Print the raw output for debugging
                    quiz_output_dict = {}
        except Exception as e:
            print(f"Error during quiz generation: {e}")
            quiz_output_dict = {}

    # If quiz output is empty or doesn't contain valid data, handle gracefully
    if not quiz_output_dict:
        print("Error: Generated quiz output is empty or invalid.")
        quiz = ""
    else:
        # Extract the generated quiz from the output
        quiz = json.dumps(quiz_output_dict, indent=4)

    # Step 3: Evaluate the quiz using the second chain (passing the quiz output from step 1)
    if quiz:
        evaluation_input = {
            "subject": SUBJECT,
            "quiz": quiz
        }

        try:
            # Run the review chain to evaluate the quiz
            review_output = review_chain.run(evaluation_input)

            # Print the result (this will include both the quiz and the review)
            print("\nGenerated Quiz in DataFrame Format:")

            # Extract the quiz and represent it as a DataFrame
            quiz_dict = json.loads(quiz)  # Parse the quiz JSON again
            quiz_data = []

            # Check if the quiz data exists and format it correctly
            if "quiz" in quiz_dict:
                for idx, item in enumerate(quiz_dict["quiz"], start=1):
                    mcq = item["mcq"]
                    options = item["options"]
                    answer = item["answer"]

                    # Append each question with its options and answer to the list
                    quiz_data.append({
                        "Question": mcq,
                        "Option A": options.get('a', ''),
                        "Option B": options.get('b', ''),
                        "Option C": options.get('c', ''),
                        "Option D": options.get('d', ''),
                        "Answer": answer
                    })

                    print("\nReview of the Quiz:")
            print(review_output)

        except Exception as e:
            print(f"Error during quiz evaluation: {e}")
    else:
        print("No quiz generated, skipping evaluation.")
