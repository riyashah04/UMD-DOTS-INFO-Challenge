import openai
import pandas as pd
import logging

# Set up OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your actual OpenAI API key

# Set up logging for error identification
logging.basicConfig(filename="error_log.txt", level=logging.ERROR)

# Function to read the CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)  # Read the CSV file
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

# Function to send CSV data to ChatGPT (if needed)
def chat_with_gpt(csv_text, question="Analyze this data."):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" if needed
            messages=[
                {"role": "system", "content": "You are an AI assistant that processes CSV files."},
                {"role": "user", "content": f"Here is the CSV data:\n{csv_text}\n\n{question}"}
            ],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error with OpenAI API request: {e}")
        return None

# Dummy function to simulate AI model prediction (for parking decisions)
def model_predict(input_data):
    # Simulate a decision-making process with some logic (e.g., based on the permit type)
    if input_data['Permit'] == 'Expired':
        return "Not Allowed"
    else:
        return "Allowed"

# Function to handle parking permission request and error identification
def handle_parking_request(input_data, model):
    # Get model's prediction
    prediction = model(input_data)

    # Log the prediction
    logging.info(f"Prediction for {input_data['Customer']}: {prediction}")

    # If the model's decision is unusual (for example, predicting 'Not Allowed' for an allowed permit)
    if prediction == "Not Allowed" and input_data['Permit'] != 'Expired':
        logging.error(f"Error: {input_data['Customer']} with {input_data['Permit']} was incorrectly flagged.")
        # Send alert to DOTS or flag for manual override
        print(f"Alert: {input_data['Customer']} has been flagged. Manual review required!")
        
        # Manual Override Process (in real-world, would be a dashboard or admin input here)
        user_input = input(f"Do you want to override this decision for {input_data['Customer']}? (y/n): ")
        if user_input.lower() == 'y':
            prediction = "Allowed"  # Manual override decision
            print(f"Manual override: {input_data['Customer']} is now Allowed.")

    return prediction

# Main execution
if __name__ == "__main__":
    file_path = "parking_history.csv"  # Replace with your CSV file path
    df = read_csv_file(file_path)

    if df is not None:
        # Convert the DataFrame to string to pass to ChatGPT (optional)
        csv_text = df.to_string(index=False)
        
        # Send the data to ChatGPT (optional)
        question = input("Enter your question for the AI regarding the CSV data: ")
        response = chat_with_gpt(csv_text, question)

        if response:
            print("\nChatGPT Response:\n", response)
        
        # Apply model prediction and error handling to each customer
        for index, row in df.iterrows():
            print(f"Processing {row['Customer']}...")
            decision = handle_parking_request(row, model_predict)
            print(f"Decision: {decision}\n")
    else:
        print("Failed to load the CSV file.")
