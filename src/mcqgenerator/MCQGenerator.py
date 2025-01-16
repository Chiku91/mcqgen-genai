import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, create_and_print_dataframe
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.logger import logging

# Loading JSON file (make sure to use the correct path for your environment)
with open('C:/Users/ASUS/Desktop/mcqgen/Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Creating a title for the app
st.title("MCQs Creator Application with Langchain 🦜⛓️")

# Create a form with st.form
with st.form("user_inputs"):
    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")

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

                # Use the OpenAI callback to generate the MCQs
                with get_openai_callback() as callback:
                    # Assuming that `generate_mcqs` is a function that generates MCQs
                    response = callback(input_data)
                    
                    # Process the response (assuming response contains MCQs)
                    mcqs = response.get('mcqs', [])

                    if mcqs:
                        # Display the generated MCQs
                        st.write("Generated MCQs:")
                        create_and_print_dataframe(mcqs)
                    else:
                        st.error("No MCQs generated. Please try again.")
                    
                    # Total Tokens
                    st.write(f"Total Tokens: {callback.total_tokens}")    
                    # Input Token
                    st.write(f"Prompt Tokens: {callback.prompt_tokens}")
                    # Output Token
                    st.write(f"Completion Tokens: {callback.completion_tokens}")
                    # Total Cost
                    st.write(f"Total Cost: {callback.total_cost}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error: {traceback.format_exc()}")

# Assuming quiz_dict is provided elsewhere in the code
quiz_dict = {
    "quiz": [
        {
            "mcq": "What is the capital of France?",
            "options": {
                "a": "Paris",
                "b": "London",
                "c": "Berlin",
                "d": "Madrid"
            },
            "answer": "a"
        },
        {
            "mcq": "Which is the largest ocean?",
            "options": {
                "a": "Atlantic",
                "b": "Indian",
                "c": "Pacific",
                "d": "Arctic"
            },
            "answer": "c"
        }
    ]
}

# List to hold the quiz data
quiz_data = []

# Check if quiz_dict is a dictionary and contains the key 'quiz' with a list value
if isinstance(quiz_dict, dict) and "quiz" in quiz_dict and isinstance(quiz_dict["quiz"], list):
    # Loop through each item in the quiz list
    for idx, item in enumerate(quiz_dict["quiz"], start=1):
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
    
    # Review text area
    review = st.text_area("Please provide your review for the quiz:", height=150)

    if review:
        st.write("Your review has been submitted:")
        st.write(review)
else:
    st.error("Error in the table data")
