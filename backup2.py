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

temperature = []
humidity=[]
Winds=[]

#method extract the relevant weather features from the file
def txtCheck(array):
    #arrays to store weather features
    temperature = []
    humidity = []
    winds = []

    for i in array:

        blob = TextBlob(i)       

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

#Scratch class for KNN-MODEL
class SimpleKNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = []
        for x in X:
            #distances = np.linalg.norm(self.X_train - x, axis=1)
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_values = self.y_train[k_indices]
            y_pred.append(np.mean(k_nearest_values))
        return np.array(y_pred)

#Scratch class for GBR-MODEL 
class SimpleGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.01):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        y_pred = np.zeros_like(y, dtype=float)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            model = np.polyfit(X.flatten(), residuals, 1)
            y_pred += self.learning_rate * np.polyval(model, X.flatten())
            self.models.append(model)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=float)
        for model in self.models:
            y_pred += self.learning_rate * np.polyval(model, X.flatten())
        return y_pred

#Scratch clas for ARIMA-MODEL
class SimpleARModel:
    def __init__(self, p):
        self.p = p
        self.coef_ = None

    def fit(self, y):
        X = np.column_stack([np.roll(y, i) for i in range(1, self.p + 1)])
        X = X[self.p:]
        y = y[self.p:]
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, y, n_steps):
        predictions = []
        for _ in range(n_steps):
            X = y[-self.p:]
            pred = np.dot(X, self.coef_)
            predictions.append(pred)
            y = np.append(y, pred)
        return predictions

#Scratch class for SVM-MODEL 
class LinearSVR:
    def __init__(self, C=1e3):
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = np.dot(X, X.T)

        P = matrix(np.vstack([
            np.hstack([K, -K]),
            np.hstack([-K, K])
        ]), tc='d')
        q = matrix(self.C * np.ones(2 * n_samples) - np.hstack([y, -y]), tc='d')
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

        self.w = np.dot((alphas[:n_samples] - alphas[n_samples:]), X)
        support_indices = np.where((alphas[:n_samples] - alphas[n_samples:]) != 0)[0]
        self.b = np.mean(y[support_indices] - np.dot(X[support_indices], self.w))

    def predict(self, X):
        return np.dot(X, self.w) + self.b


#SVM-MODEL
def SVM_Model(temp):
    if temp is not None:
        history = temp.copy()
        x_train = np.array(range(len(history))).reshape(-1, 1)
        y_train = np.array(history)
        model = LinearSVR(C=1e3)
        model.fit(x_train, y_train)
        forecast = model.predict(np.array([[len(history)]]))[0]
        return forecast
    else:
        print("THE DATA NOT AVAILABLE FOR FORECASTING")

#ARIMA-MODEL
def ARIMA_MODEL(temp):
    if temp is not None:
        model = SimpleARModel(p=5)
        model.fit(np.array(temp))
        forecast = model.predict(np.array(temp), 1)[0]
        return forecast
    else:
        print("THE DATA NOT AVAILABLE FOR FORECASTING")

#GB-MODEL
def GradientBoostingModel(temp):
    if temp is not None:
        X = np.array(range(len(temp))).reshape(-1, 1)
        y = np.array(temp)
        model = SimpleGradientBoostingRegressor()
        model.fit(X, y)
        forecast = model.predict(np.array([[len(temp)]]))[0]
        return forecast
    else:
        print("THE DATA NOT AVAILABLE FOR FORECASTING")

#KNN-MODEL
def KNN_Model(temp):
    if temp is not None:
        X = np.array(range(len(temp))).reshape(-1, 1)
        y = np.array(temp)
        model = SimpleKNNRegressor()
        model.fit(X, y)
        forecast = model.predict(np.array([[len(temp)]]))[0]
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")

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
    result = txtCheck(array)
    if result is not None:
        temperature, humidity, Winds = result
        selected_model = model_dropdown.get()
        #drop-down SVM selected
        if selected_model == 'SVM Model':
            temperature1 = round(SVM_Model(temperature))
            wind1 = round(SVM_Model(Winds))
            humidity1 = round(SVM_Model(humidity))
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
            temperature2 = int(ARIMA_MODEL(temperature))
            wind2 = int(ARIMA_MODEL(Winds))
            humidity2 = int(ARIMA_MODEL(humidity))
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
            temperature3 = int(GradientBoostingModel(temperature))
            wind3 = int(GradientBoostingModel(Winds))
            humidity3 = int(GradientBoostingModel(humidity))
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
            temperature4 = int(KNN_Model(temperature))
            wind4 = int(KNN_Model(Winds))
            humidity4 = int(KNN_Model(humidity))
            title = 'K-NN Model'
            fig = display_bar_graph(title, temperature4, wind4, humidity4, ['red', 'blue', 'green'])
            top = tk.Toplevel(window)
            top.title("Model Forecast")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()
            print("\nK-NN Model")
            print("Temperature:", temperature4, "°C")
            print("Wind:", wind4, "km/h")
            print("Humidity:", humidity4, "%")
            print("\nAccording TO K-NN Model: ")
            measure(temperature4, wind4, humidity4)
            
            ##drop-down ALL selected
        elif selected_model == 'All':
            temperature1 = round(SVM_Model(temperature))
            wind1 = round(SVM_Model(Winds))
            humidity1 = round(SVM_Model(humidity))
            temperature2 = int(ARIMA_MODEL(temperature))
            wind2 = int(ARIMA_MODEL(Winds))
            humidity2 = int(ARIMA_MODEL(humidity))
            temperature3 = int(GradientBoostingModel(temperature))
            wind3 = int(GradientBoostingModel(Winds))
            humidity3 = int(GradientBoostingModel(humidity))
            temperature4 = int(KNN_Model(temperature))
            wind4 = int(KNN_Model(Winds))
            humidity4 = int(KNN_Model(humidity)) 

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
            allTemp.append(temperature4)
            allWind.append(wind4)
            allHumidty.append(humidity4)
    
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
            print("Temperature:", temperature4, "°C")
            print("Wind:", wind4, "km/h")
            print("Humidity:", humidity4, "%")
            print("\nAccording TO K-NN Model: ")
            measure(temperature4, wind4, humidity4)
            print("\n")

            # Create a grouped bar plot
            fig, ax = plt.subplots()
            labels = ['Temperature', 'Wind', 'Humidity']
            x = np.arange(len(labels))
            width = 0.2

            rects1 = ax.bar(x - 1.5*width, [temperature1, wind1, humidity1], width, label='SVM Model', color='red')
            rects2 = ax.bar(x - 0.5*width, [temperature2, wind2, humidity2], width, label='ARIMA Model', color='blue')
            rects3 = ax.bar(x + 0.5*width, [temperature3, wind3, humidity3], width, label='Gradient Boosting Model', color='green')
            rects4 = ax.bar(x + 1.5*width, [temperature4, wind4, humidity4], width, label='K-NN Model', color='yellow')

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

            temperature.clear()
            humidity.clear()
            Winds.clear()

            #Extracting last line from data because its todays weather
            res = txtCheck(lastarray)
            temperature, humidity, Winds = res

            print("\n")
            print( temperature)
            print( humidity)
            print( Winds)
            print("\n'")
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