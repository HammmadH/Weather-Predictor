import numpy as np
import pandas as pd
from textblob import TextBlob
from cvxopt import matrix, solvers
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

print("WELCOME TO WEATHER FORCASTER")

print("\n\nREAD FILE CONTENT BELOW \n\n")

last_line=''
array = []
#extract data file
with open("weather.txt", "r") as file:
    
    contents = file.read()
    array = contents.split('.')

lines = contents.splitlines()

#get last line
last_line = lines[-1].strip()
lastarray=lines[-1].split(' ')  

#print data
for i in array:
    print(i)
        

print("-------------------------------------------------------------------------------")


#method extract the relevant weather features from the file
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

#Scratch class for GBR-MODEL 
class SimpleGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.01):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        self.models = []
        y_pred = np.zeros_like(y, dtype=float)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            model = [np.polyfit(X.flatten(), residuals[:, i], 1) for i in range(residuals.shape[1])]
            y_pred += self.learning_rate * np.array([np.polyval(m, X.flatten()) for m in model]).T
            self.models.append(model)

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], len(self.models[0])), dtype=float)
        for model in self.models:
            y_pred += self.learning_rate * np.array([np.polyval(m, X.flatten()) for m in model]).T
        return y_pred

#Scratch clas for ARIMA-MODEL
class SimpleARModel:
    def __init__(self, p):
        self.p = p
        self.coefs_ = []

    def fit(self, y):
        self.coefs_ = []
        for i in range(y.shape[1]):
            yi = y[:, i]
            X = np.column_stack([np.roll(yi, j) for j in range(1, self.p + 1)])
            X = X[self.p:]
            yi = yi[self.p:]
            coef = np.linalg.lstsq(X, yi, rcond=None)[0]
            self.coefs_.append(coef)

    def predict(self, y, n_steps):
        predictions = np.zeros((n_steps, y.shape[1]))
        for step in range(n_steps):
            X = y[-self.p:]
            for i, coef in enumerate(self.coefs_):
                pred = np.dot(X[:, i], coef)
                predictions[step, i] = pred
                y = np.append(y, [[pred] * y.shape[1]], axis=0)
        return predictions

#Scratch class for SVM-MODEL 
class LinearSVR:
    def __init__(self, C=1e3):
        self.C = C
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        for i in range(y.shape[1]):
            yi = y[:, i]
            K = np.dot(X, X.T)

            P = matrix(np.vstack([
                np.hstack([K, -K]),
                np.hstack([-K, K])
            ]), tc='d')
            q = matrix(self.C * np.ones(2 * n_samples) - np.hstack([yi, -yi]), tc='d')
            G = matrix(np.vstack([
                -np.eye(2 * n_samples),
                np.eye(2 * n_samples)
            ]), tc='d')
            h = matrix(np.hstack([np.zeros(2 * n_samples), np.ones(2 * n_samples) * self.C]), tc='d')

            A = matrix(np.hstack([np.ones(n_samples), -np.ones(n_samples)]).reshape(1, -1), tc='d')
            b = matrix(0.0, tc='d')

            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            alphas = np.array(solution['x']).flatten()

            w = np.dot((alphas[:n_samples] - alphas[n_samples:]), X)
            support_indices = np.where((alphas[:n_samples] - alphas[n_samples:]) != 0)[0]
            b = np.mean(yi[support_indices] - np.dot(X[support_indices], w))

            self.models.append((w, b))

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], len(self.models)), dtype=float)
        for i, (w, b) in enumerate(self.models):
            y_pred[:, i] = np.dot(X, w) + b
        return y_pred


#SVM-MODEL
def SVM_Model(weather):
    if weather is not None and len(weather) > 0:
        X = np.array(range(len(weather))).reshape(-1, 1)
        y = np.array(weather)
        
        model = LinearSVR(C=1e3)
        model.fit(X, y)
        
        # Predicting the next value (forecasting)
        forecast = model.predict(np.array([[len(weather)]]))[0]
        
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")
        return None

#ARIMA-MODEL
def ARIMA_MODEL(weather):
    if weather is not None and len(weather) > 0:
        y = np.array(weather)
        
        model = SimpleARModel(p=5)
        model.fit(y)
        
        # Predicting the next value (forecasting)
        forecast = model.predict(y, 1)[0]
        
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")
        return None

#GB-MODEL
def GradientBoostingModel(weather):
    if weather is not None and len(weather) > 0:
        X = np.array(range(len(weather))).reshape(-1, 1)
        y = np.array(   weather)
        
        model = SimpleGradientBoostingRegressor()
        model.fit(X, y)
        
        # Predicting the next value (forecasting)
        forecast = model.predict(np.array([[len(weather)]]))[0]
        
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")
        return None

def _euclidean_distance(x1, x2):
    print(x1, x2)
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN-MODEL
def KNN_Model(x_train, y_train, X_test, k):
    y_pred = []
    distances = []
    for train_point in x_train:        
        distance = _euclidean_distance(X_test, train_point)
        distances.append(distance)
        
    k_indices = np.argsort(distances)[:k]
    k_nearest_values = y_train[k_indices]
    prediction = np.mean(k_nearest_values, axis=0)
    y_pred.append(prediction)



    # for test_point in X_test:
    #     distances = []
    #     for train_point in x_train:
    #         distance = _euclidean_distance(test_point, train_point)
    #         distances.append(distance)
        
    #     k_indices = np.argsort(distances)[:k]
    #     k_nearest_values = y_train[k_indices]
    #     prediction = np.mean(k_nearest_values, axis=0)
    #     y_pred.append(prediction)

    return np.array(y_pred)

# Function to create and display bar graph
def display_bar_graph(title, temperature, wind, humidity, colors):
    labels = ['Temperature', 'Wind', 'Humidity']
    values = [temperature, wind, humidity]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors, width=0.4)
    ax.set_title(title)
    ax.bar_label(bars)

    return fig
    
#Function to check weather of a paticular model and all selected from drop-down menu
def check_weather():
    #initialize all arrays to store every model data
    allTemp=[]
    allHumidty=[]
    allWind=[]

    weather_data = txtCheck(array)
    print("Weather Data:", weather_data)

    if weather_data is not None:
        selected_model = model_dropdown.get()
        #drop-down SVM selected
        if selected_model == 'SVM Model':
            forecast = SVM_Model(weather_data)
            temperature1 = int(forecast[0])
            humidity1 = int(forecast[1])
            wind1 = int(forecast[2])
            title = 'SVM Model'
            fig = display_bar_graph(title, temperature1, wind1, humidity1, ['red', 'blue', 'green'])
            top = tk.Toplevel(window)
            top.title("Model Forecast")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()
            print("\nSVM Model")
            print("Temperature:", temperature1, "°C")
            print("Wind:", wind1, "km/h")
            print("Humidity:", humidity1, "%")
            print("\nAccording TO SVM Model: ")
            measure(temperature1, wind1, humidity1)
            
            #drop-down ARIMA selected
        elif selected_model == 'ARIMA Model':
            forecast = ARIMA_MODEL(weather_data)
            temperature2 = int(forecast[0])
            humidity2 = int(forecast[1])
            wind2 = int(forecast[2])
            title = 'ARIMA Model'
            fig = display_bar_graph(title, temperature2, wind2, humidity2, ['red', 'blue', 'green'])
            top = tk.Toplevel(window)
            top.title("Model Forecast")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()
            print("\nARIMA Model")
            print("Temperature:", temperature2, "°C")
            print("Wind:", wind2, "km/h")
            print("Humidity:", humidity2, "%")
            print("\nAccording TO ARIMA Model: ")
            measure(temperature2, wind2, humidity2)
            
            #drop-down GBM selected
        elif selected_model == 'Gradient Boosting Model':
            forecast = GradientBoostingModel(weather_data)
            temperature3 = int(forecast[0])
            humidity3 = int(forecast[1])
            wind3 = int(forecast[2])
            title = 'Gradient Boosting Model'
            fig = display_bar_graph(title, temperature3, wind3, humidity3, ['red', 'blue', 'green'])
            top = tk.Toplevel(window)
            top.title("Model Forecast")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()
            print("\nGradient Boosting Model")
            print("Temperature:", temperature3, "°C")
            print("Wind:", wind3, "km/h")
            print("Humidity:", humidity3, "%")
            print("\nAccording TO Gradient Boosting Model: ")
            measure(temperature3, wind3, humidity3)
            
            #drop-down KNN selected
        elif selected_model == 'K-NN Model':
            x_train = np.array(weather_data)
            y_train = x_train.copy()

            res = ltxtCheck(lastarray)
            x_test = np.array([res])
            forecast = KNN_Model(x_train, y_train, x_test, 5)

            temperature4, humidity4, wind4 = forecast[0]
            title = 'K-NN Model'
            fig = display_bar_graph(title, int(temperature4), int(wind4), int(humidity4), ['red', 'blue', 'green'])
            top = tk.Toplevel(window)
            top.title("Model Forecast")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()
            print("\nK-NN Model")
            print("Temperature:", int(temperature4), "°C")
            print("Wind:", int(wind4), "km/h")
            print("Humidity:", int(humidity4), "%")
            print("\nAccording TO K-NN Model: ")
            measure(int(temperature4), int(wind4), int(humidity4))
            
            ##drop-down ALL selected
        elif selected_model == 'All':
            forecast1 = SVM_Model(weather_data)
            temperature1 = int(forecast1[0])
            humidity1 = int(forecast1[1])
            wind1 = int(forecast1[2])
            forecast2 = ARIMA_MODEL(weather_data)
            temperature2 = int(forecast2[0])
            humidity2 = int(forecast2[1])
            wind2 = int(forecast2[2])
            forecast3 = GradientBoostingModel(weather_data)
            temperature3 = int(forecast3[0])
            humidity3 = int(forecast3[1])
            wind3 = int(forecast3[2])
            x_train = np.array(weather_data)
            y_train = x_train.copy()
            res = ltxtCheck(lastarray)
            x_test = np.array([res])
            forecast4 = KNN_Model(x_train, y_train, x_test, 5)
            temperature4, humidity4, wind4 = forecast4[0] 

            #storing data in all arrays
            allTemp.append(temperature1)
            allWind.append(wind1)
            allHumidty.append(humidity1)
            allTemp.append(temperature2)
            allWind.append(wind2)
            allHumidty.append(humidity2)
            allTemp.append(temperature3)
            allWind.append(wind3)
            allHumidty.append(humidity3)
            allTemp.append(int(temperature4))
            allWind.append(int(wind4))
            allHumidty.append(int(humidity4))
    
            print("\nComparative Analysis:")
            print("\nSVM Model")
            print("Temperature:", temperature1, "°C")
            print("Wind:", wind1, "km/h")
            print("Humidity:", humidity1, "%")
            print("\nAccording TO SVM Model: ")
            measure(temperature1, wind1, humidity1)
            print("\nARIMA Model")
            print("Temperature:", temperature2, "°C")
            print("Wind:", wind2, "km/h")
            print("Humidity:", humidity2, "%")
            print("\nAccording TO ARIMA Model: ")
            measure(temperature2, wind2, humidity2)
            print("\nGradient Boosting Model")
            print("Temperature:", temperature3, "°C")
            print("Wind:", wind3, "km/h")
            print("Humidity:", humidity3, "%")
            print("\nAccording TO Gradient Boosting Model: ")
            measure(temperature3, wind3, humidity3)
            print("\nK-NN Model")
            print("Temperature:", int(temperature4), "°C")
            print("Wind:", int(wind4), "km/h")
            print("Humidity:", int(humidity4), "%")
            print("\nAccording TO K-NN Model: ")
            measure(int(temperature4), int(wind4), int(humidity4))
            print("\n")

            # Create a grouped bar plot
            fig, ax = plt.subplots()
            labels = ['Temperature', 'Wind', 'Humidity']
            x = np.arange(len(labels))
            width = 0.2

            rects1 = ax.bar(x - 1.5*width, [temperature1, wind1, humidity1], width, label='SVM Model', color='red')
            rects2 = ax.bar(x - 0.5*width, [temperature2, wind2, humidity2], width, label='ARIMA Model', color='blue')
            rects3 = ax.bar(x + 0.5*width, [temperature3, wind3, humidity3], width, label='Gradient Boosting Model', color='green')
            rects4 = ax.bar(x + 1.5*width, [int(temperature4), int(wind4), int(humidity4)], width, label='K-NN Model', color='yellow')

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title('Comparison of ALL Models')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)
            ax.bar_label(rects3, padding=3)
            ax.bar_label(rects4, padding=3)

            fig.tight_layout()
            top = tk.Toplevel(window)
            top.title("All Models Forecast")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()
            
            #finding max of data
            MaxTemperature=allTemp.index(max(allTemp))
            MaxHumidty=allHumidty.index(max(allHumidty))
            MaxWinds=allWind.index(max(allWind))

            temp=allTemp[MaxTemperature]
            humid=allHumidty[MaxHumidty]
            windss=allWind[MaxWinds]

            print("\nMAXIMUM\n")
            print(int(temp),"°C Temperature")
            print(int(humid),"% Humdity")
            print(int(windss)," km/h Winds\n")

            #Extracting last line from data because its todays weather
            res = ltxtCheck(lastarray)
            temperature, humidity, Winds = res

            print("\n")
            print( temperature)
            print( humidity)
            print( Winds)
            print("\n")
            print("Today is ",last_line)
            print("\n")
            print( allTemp)
            print( allHumidty)
            print( allWind)

            model1count=0
            model2count=0
            model3count=0
            model4count=0

            print("\n================== MODEL SATISFACTION CHECK ==================")

            while True:
                if allTemp[0]==temperature[0]:
                    print("Temperature Satisfied By SVM MODEL")
                    model1count+=1
                    if allWind[0]==Winds[0]:
                        print("Wind Satisfied By SVM MODEL")
                        model1count+=1
                        if allHumidty[0]==humidity[0]:
                            print("Humidity Satisfied By SVM MODEL")
                            model1count+=1
            
                elif allTemp[1]==temperature[0]:
                    print("Temperature Satisfied By ARIMA MODEL")
                    model2count+=1
                    if allWind[1]==Winds[0]:
                        print("Wind Satisfied By ARIMA MODEL")
                        model2count+=1
                        if allHumidty[1]==humidity[0]:
                            print("Humidity Satisfied By ARIMA MODEL")
                            model2count+=1

                elif allTemp[2]==temperature[0]:
                    print("Temperature Satisfied By GRADIENT BOOSTING MODEL")
                    model3count+=1
                    if allWind[2]==Winds[0]:
                        print("Wind Satisfied By GRADIENT BOOSTING MODEL")
                        model3count+=1
                        if allHumidty[2]==humidity[0]:
                            print("Humidity Satisfied By GRADIENT BOOSTING MODEL")
                            model3count+=1

                elif allTemp[3]==temperature[0]:
                    print("Temperature Satisfied By k-NN Model")
                    model4count+=1
                    if allWind[3]==Winds[0]:
                        print("Wind Satisfied By k-NN Model")
                        model4count+=1
                        if allHumidty[3]==humidity[0]:
                            print("Humidity Satisfied By k-NN Model")
                            model4count+=1

                print(model1count, model2count, model3count, model4count)

                if model1count==3:
                    print("\nSVM MODEL PREDICTING WEATHER CORRECTLY")
                    break   

                if model2count==3:
                    print("\nARIMA MODEL PREDICTING WEATHER CORRECTLY")
                    break   

                if model3count==3:
                    print("\nGRADIENT MODEL PREDICTING WEATHER CORRECTLY")
                    break

                if model4count==3:
                    print("\nKN-N MODEL PREDICTING WEATHER CORRECTLY")
                    break 
                
    
                else:
                    print("\nModels gave us the approximate Prediction but not accurate. ")

                    if model1count > model2count or model1count > model3count or model1count > model4count:
                        print("SVM MODEL PREDICTING THE RESULT BETTER AS COMPARE TO GRADIENT, ARIMA AND KN-N")
                    if model2count > model1count or model2count > model3count or model2count > model4count:
                        print("ARIMA MODEL PREDICTING THE RESULT BETTER AS COMPARE TO GRADIENT, SVM AND KN-N")
                    if model3count > model2count or model3count > model1count or model3count > model4count:
                        print("GRADIENT BOOSTING MODEL PREDICTING THE RESULT BETTER AS COMPARE TO SVM, ARIMA AND KN-N")
                    if model4count > model1count or model4count > model2count or model4count > model3count:
                        print("KN-N MODEL PREDICTING THE RESULT BETTER AS COMPARE TO SVM, ARIMA AND GRADIENT")
                    break
    else:
        print("No data available for weather forecasting")    
       
#Weather Conditions statements
def measure(temp,windss,humid):
    if (temp> 25):
        if (windss >=15):
            if(humid>40 and humid <70):
                 print("The weather would be Hot and Sunny.")
                 print("The chances of rain are low.")
                 return
            else:
                 print("The weather would be Hot and Sunny.")
                 print("The chances of rain are lil bit HIGH.")
                 return
            
    elif (temp > 15 and temp < 25) and (windss >= 5 and windss < 20) and (humid >= 70):
        print("It is a mild temperature, moderate wind, and relatively humid.")
        print("The chances of rain are moderate.")
        return

    elif (temp <= 15) and (windss >= 20) and (humid >= 70):
        print("The weather would be cold, windy, and have high humidity.")
        print("The chances of rain are high.")
        return
    
    elif (temp <= 15) and (windss >= 20):
        print("The weather would be cold and windy.")
        print("The chances of rain are low to moderate.")
        return
    
    elif (temp <= 15) and (humid >= 70):
        print("The weather would be cold and humid.")
        print("The chances of rain are moderate to high.")
        return
    
    elif (windss >= 20) and (humid >= 70):
        print("The weather would be windy and humid.")
        print("The chances of rain are moderate to high.")
        return
    
    elif (temp > 25) and (windss <= 5) and (40 <= humid <= 70):
        print("The weather would be Hot with moderate humidity.")
        print("The chances of rain are low to moderate.")
        return
    
    else:
        print("No specific condition matched for weather or rain chances.")
        return
    
    
# GUI Setup
window = tk.Tk()
window.title("Weather Forecaster")
window.geometry("400x400")

# Background Image
background_image = tk.PhotoImage(file="bgg.png")
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

font_heading = ("Helvetica", 16, "bold")
font_normal = ("Helvetica", 12)

heading = tk.Label(window, text="WEATHER FORECASTER", font=font_heading)
heading.pack(pady=10)

# Model Dropdown
model_label = tk.Label(window, text="Select Forecasting Model:", font=font_normal)
model_label.pack(pady=10)
model_options = ['SVM Model', 'ARIMA Model', 'Gradient Boosting Model', 'K-NN Model', 'All']
model_dropdown = ttk.Combobox(window, values=model_options)
model_dropdown.pack()

# Weather Check Button
check_button = tk.Button(window, text="Forecast", command=check_weather, font=font_heading)
check_button.pack(pady=10)

window.mainloop()