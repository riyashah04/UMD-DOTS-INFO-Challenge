import openai
import pandas as pd

# Set up your OpenAI API key
client = openai.OpenAI(api_key="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")  # Replace with your actual OpenAI API key

# Function to read the CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)  # Read the CSV file
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

# Function to send CSV data to ChatGPT
def chat_with_gpt(csv_text, question="Analyze this data."):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" if needed
            messages=[
                {"role": "system", "content": "You are an AI assistant that processes CSV files."},
                {"role": "user", "content": f"Here is the CSV data:\n{csv_text}\n\n{question}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI API request: {e}")
        return None

# Main execution
if __name__ == "__main__":
    file_path = "Data.csv"  # Replace with your CSV file path
    df = read_csv_file(file_path)

    if df is not None:
        csv_text = df.to_string(index=False)  # Convert DataFrame to a string
        question = input("Enter your question for the AI regarding the CSV data: ")
        response = chat_with_gpt(csv_text, question)

        if response:
            print("\nChatGPT Response:\n", response)

