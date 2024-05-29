import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkcalendar import Calendar
import numpy as np
import datetime
from tensorflow.keras.models import load_model

# Function to load the MSFT dataset
import pandas as pd

def load_data():
    try:
        # Load the MSFT dataset from a CSV file
        # Replace 'path_to_msft_data.csv' with the actual path to your dataset file
        df = pd.read_csv('MSFT.csv')
        

        # Assuming the dataset contains 'Close' prices
        close_prices = df['Close'].values

        # Reshape the data to match the input shape of the model
        X = np.reshape(close_prices, (-1, 3, 1))

        return X
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")
        return None


# Function to predict stock prices using the loaded model
def predict(model, X):
    # Make predictions
    predictions = model.predict(X).flatten()
    return predictions

# Function to handle predict button click event
def on_predict():
    global model, calendar_start, calendar_end, label_result

    # Get selected dates
    date_start = calendar_start.get_date()
    date_end = calendar_end.get_date()

    # Convert selected dates to datetime objects
    start_date = datetime.datetime.strptime(date_start, '%m/%d/%y')
    end_date = datetime.datetime.strptime(date_end, '%m/%d/%y')

    # Load data and preprocess
    X = load_data()

    # Predict
    predictions = predict(model, X)

    # Display result
    result_str = "Predictions:\n"
    for i, pred in enumerate(predictions):
        result_str += f"Prediction {i+1}: {pred}\n"

    label_result.config(text=result_str)

# Function to load the pre-trained stockforecasting model
def load_stockforecasting_model():
    try:
        # Load the pre-trained model named "stockforecasting.keras"
        model = load_model('stockforecasting.keras')
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return None

# Create the main window
root = tk.Tk()
root.title("Stock Price Prediction")

# Load the stockforecasting model
model = load_stockforecasting_model()

# Calendar for selecting start date
label_start = tk.Label(root, text="Select Start Date:")
label_start.pack()
calendar_start = Calendar(root, selectmode="day", date_pattern="mm/dd/yy")
calendar_start.pack()

# Calendar for selecting end date
label_end = tk.Label(root, text="Select End Date:")
label_end.pack()
calendar_end = Calendar(root, selectmode="day", date_pattern="mm/dd/yy")
calendar_end.pack()

# Predict button
predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack()

# Result label
label_result = tk.Label(root, text="")
label_result.pack()

# Run the Tkinter event loop
root.mainloop()
