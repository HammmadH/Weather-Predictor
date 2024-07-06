import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Weatherpredictor as wp




# GUI Setup
window = tk.Tk()
window.title("Weather Forecast")
window.geometry("600x400")

# Background Image
background_image = tk.PhotoImage(file="bg.png")  # Replace with your background image path
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Model Dropdown
model_label = tk.Label(window, text="Select Model:")
model_label.pack(pady=10)
model_options = ['SVM Model', 'ARIMA Model', 'Gradient Boosting Model', 'K-NN Model', 'All']
model_dropdown = ttk.Combobox(window, values=model_options)
model_dropdown.pack()

# Weather Check Button
check_button = tk.Button(window, text="Check Weather", command=wp.check_weather)
check_button.pack(pady=10)

window.mainloop()