import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from textblob import TextBlob

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


print("WELCOME TO WEATHER FORCASTER BY USING PREPOSITIONAL LOGICS")

print("\n\nREAD FILE CONTENT BELOW \n\n")

# ctrl F5
last_line=''
# Open the text file
array = []
with open("weather.txt", "r") as file:
    # Read the content of the file
    
    contents = file.read()
    array = contents.split('.')

lines = contents.splitlines()

# Get the last line
last_line = lines[-1].strip()
lastarray=lines[-1].split(' ')  


#PRINTING THE CONTENT
for i in array:
    
    print(i)
        

print("-------------------------------------------------------------------------------")

temperature = []
humidity=[]
Winds=[]


def txtCheck(array):
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



# =======================================================================================

#ALGORITHMS


def SVM_Model(temp):
    if temp is not None:
        # TODO: Replace with actual data and hyperparameters
        history = temp.copy()
        x_train = np.array(range(len(history))).reshape(-1, 1)
        y_train = np.array(history)
        model = SVR(kernel='linear', C=1e3)
        model.fit(x_train, y_train)
        forecast = model.predict(np.array(len(history)).reshape(-1, 1))[0]
        return forecast
    else:
        print("THE DATA NOT AVAIALABLE FOR FORECASTING")

def ARIMA_MODEL(temp):
    #Autoregressive Integrated Moving Average
    #forcast on future values
    if temp is not None:
        # TODO: Replace with actual data and hyperparameters
        history = temp.copy()
        model = ARIMA(history, order=(1,0,5))
        model_fit = model.fit()
        forecast = model_fit.forecast()
        return forecast
    else:
        print("THE DATA NOT AVAIALABLE FOR FORECASTING")

def GradientBoostingModel(temp):
    #this algo is used for ensemble learning method 
    if temp is not None:
        # TODO: Replace with actual data and hyperparameters
        X = [[i] for i in range(len(temp))]
        y = temp.copy()
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=1, random_state=0)
        model.fit(X, y)
        forecast = model.predict([[len(temp)]])
        return forecast
    else:
        print("THE DATA NOT AVAIALABLE FOR FORECASTING")


# Function to train and predict using k-NN algorithm
def KNN_Model(temp):
    if temp is not None:
        # TODO: Replace with actual data and hyperparameters
        history = temp.copy()
        X = [[i] for i in range(len(history))]
        y = history
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X, y)
        forecast = model.predict([[len(history)]])
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")



#######################

# Function to create and display bar graph
def display_bar_graph(title, temperature, wind, humidity, colors):
    labels = ['Temperature', 'Wind', 'Humidity']
    values = [temperature, wind, humidity]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors, width=0.4)
    ax.set_title(title)
    ax.bar_label(bars)

    return fig
    

def check_weather():
    allTemp=[]
    allHumidty=[]
    allWinder=[]
    result = txtCheck(array)
    if result is not None:
        temperature, humidity, Winds = result
        selected_model = model_dropdown.get()
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
            
            allTemp.append(temperature1)
            allWinder.append(wind1)
            allHumidty.append(humidity1)
            allTemp.append(temperature2)
            allWinder.append(wind2)
            allHumidty.append(humidity2)
            allTemp.append(temperature3)
            allWinder.append(wind3)
            allHumidty.append(humidity3)
            allTemp.append(temperature4)
            allWinder.append(wind4)
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
            
            MaxTemperature=allTemp.index(max(allTemp))
            MaxHumidty=allHumidty.index(max(allHumidty))
            MaxWinds=allWinder.index(max(allWinder))

            temp=allTemp[MaxTemperature]
            humid=allHumidty[MaxHumidty]
            windss=allWinder[MaxWinds]

            print("\nMAXIMUM\n")
            print(int(temp),"°C Temperature")
            print(int(humid),"% Humdity")
            print(int(windss)," km/h Winds\n")

            temperature.clear()
            humidity.clear()
            Winds.clear()
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
            print( allWinder)

            model1count=0
            model2count=0
            model3count=0
            model4count=0

            print("\n================== MODEL SATISFACTION CHECK ==================")

            while True:
                if allTemp[0]==temperature[0]:
                    print("Temperature Satisfied By SVM MODEL")
                    model1count+=1
                    if allWinder[0]==Winds[0]:
                        print("Wind Satisfied By SVM MODEL")
                        model1count+=1
                        if allHumidty[0]==humidity[0]:
                            print("Humidity Satisfied By SVM MODEL")
                            model1count+=1
            
                elif allTemp[1]==temperature[0]:
                    print("Temperature Satisfied By ARIMA MODEL")
                    model2count+=1
                    if allWinder[1]==Winds[0]:
                        print("Wind Satisfied By ARIMA MODEL")
                        model2count+=1
                        if allHumidty[1]==humidity[0]:
                            print("Humidity Satisfied By ARIMA MODEL")
                            model2count+=1

                elif allTemp[2]==temperature[0]:
                    print("Temperature Satisfied By GRADIENT BOOSTING MODEL")
                    model3count+=1
                    if allWinder[2]==Winds[0]:
                        print("Wind Satisfied By GRADIENT BOOSTING MODEL")
                        model3count+=1
                        if allHumidty[2]==humidity[0]:
                            print("Humidity Satisfied By GRADIENT BOOSTING MODEL")
                            model3count+=1

                elif allTemp[3]==temperature[0]:
                    print("Temperature Satisfied By k-NN Model")
                    model4count+=1
                    if allWinder[3]==Winds[0]:
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
       
#Weather Conditions
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
window.geometry("600x400")

# Background Image
background_image = tk.PhotoImage(file="bg.png")
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Model Dropdown
model_label = tk.Label(window, text="Select Forecasting Model:")
model_label.pack(pady=10)
model_options = ['SVM Model', 'ARIMA Model', 'Gradient Boosting Model', 'K-NN Model', 'All']
model_dropdown = ttk.Combobox(window, values=model_options)
model_dropdown.pack()

# Weather Check Button
check_button = tk.Button(window, text="Forecast", command=check_weather)
check_button.pack(pady=10)

window.mainloop()