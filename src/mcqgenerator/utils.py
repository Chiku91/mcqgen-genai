import os
import PyPDF2
import json
import traceback

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader=PyPDF2.PdfFileReader(file)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            return text
            
        except Exception as e:
            raise Exception("error reading the PDF file")
        
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    
    else:
        raise Exception(
            "unsupported file format only pdf and text file suppoted"
            )
    
def create_and_print_dataframe(quiz_data):
    """
    Takes the quiz_data (list of dictionaries) and returns a pandas DataFrame.
    Also prints the DataFrame.
    """
    # Create a DataFrame from the quiz data
    df = pd.DataFrame(quiz_data)

    # Print the DataFrame
    print("\nGenerated Quiz in DataFrame Format:")
    print(df)

    return df