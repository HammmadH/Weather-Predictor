import numpy as np
import pandas as pd
from textblob import TextBlob
from cvxopt import matrix, solvers
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

print("WELCOME TO WEATHER FORECASTER")

print("\n\nREAD FILE CONTENT BELOW \n\n")

# Extract data from file
with open("weather.txt", "r") as file:
    contents = file.read()

array = contents.split('.')
lines = contents.splitlines()

# Get last line
last_line = lines[-1].strip()
lastarray = lines[-1].split(' ')  

# Print data
for i in array:
    print(i)
        
print("-------------------------------------------------------------------------------")

# Method to extract the relevant weather features from the file
def txtCheck(array):
    print(array)
    weather = []

    for i in array:
        temp_data = None
        humidity_data = None
        wind_data = None

        blob = TextBlob(i)

        for word, tag in blob.tags:
            if tag == 'CD' and '°C' in word:
                temp_data = int(word.replace('°C', '').replace('Â', ''))
            if 'Humidity:' in word:
                try:
                    humidity_data = int(word.replace('Humidity:', ''))
                except ValueError:
                    print(f"Invalid humidity value: {word}")
            if tag == 'CD' and 'km/h' in word:
                wind_data = int(word.replace('km/h', ''))

        # Append only if all data is available
        if temp_data is not None and humidity_data is not None and wind_data is not None:
            weather.append([temp_data, humidity_data, wind_data])

        print(weather)
            
    return weather

def ltxtCheck(array):
    temperature = []
    humidity = []
    winds = []

    # Create a TextBlob object
    for i in array:
        blob = TextBlob(i)
        
        # Extract the relevant weather features from the text
        for word, tag in blob.tags:
            if tag == 'CD' and '°C' in word:
                temperature.append(int(word.replace('°C', '').replace('Â', '')))
            if 'Humidity:' in word:
                try:
                    humidity.append(int(word.replace('Humidity:', '')))
                except ValueError:
                    print(f"Invalid humidity value: {word}")
            if tag == 'CD' and 'km/h' in word:
                winds.append(int(word.replace('km/h', '')))

    return temperature, humidity, winds

def _euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN-MODEL
def KNN_Model(x_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = []
        for train_point in x_train:
            distance = _euclidean_distance(test_point, train_point)
            distances.append(distance)
        
        k_indices = np.argsort(distances)[:k]
        k_nearest_values = y_train[k_indices]
        prediction = np.mean(k_nearest_values, axis=0)
        y_pred.append(prediction)

    return np.array(y_pred)

weather_data = txtCheck(array)
print("Weather Data:", weather_data)

x_train = np.array(weather_data)
y_train = x_train.copy()

res = ltxtCheck(lastarray)
x_test = np.array([res])
forecast = KNN_Model(x_train, y_train, x_test, 5)

temperature4, humidity4, wind4 = forecast[0]

print("\nK-NN Model")
print("Temperature:", int(temperature4), "°C")
print("Wind:", int(wind4), "km/h")
print("Humidity:", int(humidity4), "%")
