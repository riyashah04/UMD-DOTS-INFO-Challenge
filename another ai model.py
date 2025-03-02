import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv("parking_history.csv")

# Step 2: Print out the column names to verify them
print("Column names in the dataset:", df.columns)

# Step 3: Trim spaces from column names (in case there are leading/trailing spaces)
df.columns = df.columns.str.strip()

# Step 4: Handle non-time values in the 'Time' column
def convert_time(time_str):
    try:
        # Try to convert to datetime with the format '%H:%M'
        return pd.to_datetime(time_str, format='%H:%M').hour
    except ValueError:
        # If conversion fails (e.g., "Daily"), return a default hour (e.g., 0 for 00:00)
        return 0  # You can change this to a different default time if needed

if "Time" in df.columns:
    df['Time'] = df['Time'].apply(convert_time)  # Apply the conversion function
else:
    print("Error: 'Time' column is missing from the dataset.")

# Step 5: Encode categorical columns (assuming they exist in the dataset)
encoder = LabelEncoder()

# Ensure 'Permit' and 'Personnel Type (Fac/Staff, Student, Visitor)' columns exist
if "Permit" in df.columns and "Personnel Type (Fac/Staff, Student, Visitor)" in df.columns:
    df["Permit"] = encoder.fit_transform(df["Permit"])  # Converts "Full-time", "Temporary" to numbers
    df["PersonnelType"] = encoder.fit_transform(df["Personnel Type (Fac/Staff, Student, Visitor)"])  # Converts "Student", "Faculty", etc. to numbers
else:
    print("Error: 'Permit' or 'Personnel Type (Fac/Staff, Student, Visitor)' column is missing from the dataset.")

# Step 6: Check and identify the correct permissions column
# Print the column names to see which one represents the permissions or parking eligibility
print("Column names after trimming spaces:", df.columns)

# Assuming 'Eligible Lots (Not Exhaustive for some examples)' represents the parking eligibility
if "Eligible Lots (Not Exhaustive for some examples)" in df.columns:
    # Drop non-relevant columns
    df = df.drop(columns=["Customer", "Date", "Eligible Lots (Not Exhaustive for some examples)"])  # Drop columns that are not needed for features
    X = df  # Features: All remaining columns
    y = df["Eligible Lots (Not Exhaustive for some examples)"]  # Labels: Whether parking is allowed or not
else:
    print("Error: The permissions column is missing or misnamed.")

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # First hidden layer
    tf.keras.layers.Dense(32, activation="relu"),  # Second hidden layer
    tf.keras.layers.Dense(1, activation="sigmoid")  # Output layer (1 for Allowed, 0 for Not Allowed)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Step 9: Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 10: Evaluate the model's performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 11: Making Predictions
def predict_parking(permit, time, personnel_type):
    # Prepare input data (assuming these values are already encoded)
    input_data = pd.DataFrame([[permit, time, personnel_type]],
                              columns=["Permit", "Time", "PersonnelType"])
    
    # Make prediction
    prediction = model.predict(input_data)
    return "Allowed to park" if prediction[0] > 0.5 else "Not allowed to park"

# Example usage:
print(predict_parking(1, 10, 0))  # Example: Permit=1 (Full-time), Time=10 (10:00 AM), PersonnelType=0 (Student)

# Step 12: Save the model
model.save("parking_model.h5")

# Step 13: Load the model (if needed later)
# model = tf.keras.models.load_model("parking_model.h5")