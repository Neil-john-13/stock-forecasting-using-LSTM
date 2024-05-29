import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkcalendar import Calendar
import numpy as np
import datetime
from tensorflow.keras.models import load_model


import pandas as pd

def load_data():
    try:
       
        df = pd.read_csv('MSFT.csv')
        close_prices = df['Close'].values
        X = np.reshape(close_prices, (-1, 3, 1))

        return X
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")
        return None

def predict(model, X):
    predictions = model.predict(X).flatten()
    return predictions

def on_predict():
    global model, calendar_start, calendar_end, label_result

    date_start = calendar_start.get_date()
    date_end = calendar_end.get_date()

    start_date = datetime.datetime.strptime(date_start, '%m/%d/%y')
    end_date = datetime.datetime.strptime(date_end, '%m/%d/%y')

    X = load_data()

    predictions = predict(model, X)

    result_str = "Predictions:\n"
    for i, pred in enumerate(predictions):
        result_str += f"Prediction {i+1}: {pred}\n"

    label_result.config(text=result_str)

def load_stockforecasting_model():
    try:

        model = load_model('stockforecasting.keras')
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return None

root = tk.Tk()
root.title("Stock Price Prediction")

model = load_stockforecasting_model()

label_start = tk.Label(root, text="Select Start Date:")
label_start.pack()
calendar_start = Calendar(root, selectmode="day", date_pattern="mm/dd/yy")
calendar_start.pack()

label_end = tk.Label(root, text="Select End Date:")
label_end.pack()
calendar_end = Calendar(root, selectmode="day", date_pattern="mm/dd/yy")
calendar_end.pack()

predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack()

label_result = tk.Label(root, text="")
label_result.pack()

root.mainloop()
