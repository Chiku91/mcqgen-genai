import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, create_and_print_dataframe
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.mcqgenerator.logger import logging

# Loading environment variables
load_dotenv()  # Ensure the .env file is loaded

# Ensure the OpenAI API Key is loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key is missing. Please check your .env file.")

# Create a title for the app
st.title("MCQs Creator Application with Langchain 🦜⛓️")

# Create a form with st.form
with st.form("user_inputs"):
    # File Upload
    uploaded_file = st.file_uploader("Upload a txt file")

    # Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)

    # Subject
    subject = st.text_input("Insert Subject", max_chars=20)

    # Quiz Tone
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")

    # Add Button
    button = st.form_submit_button("Create MCQs")

    # Check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                # Read the uploaded file
                text = read_file(uploaded_file)

                # Now get the input data and pass it to the OpenAI callback
                input_data = {
                    "text": text,  # Read text from file dynamically
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone
                }

                # Define the prompt template for generating MCQs
                quiz_generation_prompt = """
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
                """

                # Set up the LLM (OpenAI model)
                llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.7)

                # Create the PromptTemplate
                prompt_template = PromptTemplate(input_variables=["text", "number", "subject", "tone"], template=quiz_generation_prompt)

                # Create an LLMChain (This is the process where you generate MCQs using OpenAI's model)
                quiz_chain = LLMChain(llm=llm, prompt=prompt_template)

                # Generate MCQs using the chain
                quiz_output = quiz_chain.invoke(input_data)

                # Check if the 'text' field is the one containing the quiz data
                try:
                    if isinstance(quiz_output, dict) and "text" in quiz_output:
                        # Extract the MCQ JSON text from the 'text' field
                        quiz_text = quiz_output["text"]
                        quiz_output_dict = json.loads(quiz_text)  # Parse the quiz data
                    else:
                        raise ValueError("Unexpected format from OpenAI response.")

                    # Extract MCQs from the parsed response
                    mcqs = quiz_output_dict.get("quiz", [])

                    if mcqs:
                        # Display the generated MCQs
                        st.write("Generated MCQs:")
                        
                        # Create DataFrame for MCQs
                        quiz_data = []
                        for item in mcqs:
                            mcq = item.get("mcq", "")
                            options = item.get("options", {})
                            answer = item.get("answer", "")

                            # Append each question with its options and answer to the list
                            quiz_data.append({
                                "Question": mcq,
                                "Option A": options.get('a', ''),
                                "Option B": options.get('b', ''),
                                "Option C": options.get('c', ''),
                                "Option D": options.get('d', ''),
                                "Answer": answer
                            })

                        # Create a DataFrame from the quiz data
                        df = pd.DataFrame(quiz_data)

                        # Display the DataFrame as a table in Streamlit
                        st.table(df)  # Use Streamlit's st.table to display the table

                        # Generate the review about the quiz
                        review_prompt = """
                        Based on the following multiple choice questions, generate a review that is about 150 words in length, 
                        summarizing the quality of the questions, the variety of the content, and the suitability for the given subject and tone.
                        Review the questions objectively:
                        {quiz}
                        """
                        
                        review_data = {
                            "quiz": json.dumps(mcqs)  # Passing the MCQs as a JSON string to OpenAI
                        }

                        review_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["quiz"], template=review_prompt))
                        review_output = review_chain.invoke(review_data)

                        # Extract only the review text (remove the surrounding JSON structure)
                        review_text = review_output.get("text", "").strip()

                        # Display the generated review (plain text)
                        st.write("Generated Review:")
                        st.text_area("Review", value=review_text, height=200, disabled=True)

                    else:
                        st.error("No MCQs generated. Please try again.")

                except Exception as e:
                    st.error(f"An error occurred while parsing the response: {str(e)}")
                    logging.error(f"Error: {traceback.format_exc()}")

                # If using callback for token details, use callback attributes like:
                if hasattr(quiz_chain.llm, 'get_tokens_used'):
                    # If tokens are available, display token details
                    st.write(f"Total Tokens: {quiz_chain.llm.get_tokens_used()}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error: {traceback.format_exc()}")
