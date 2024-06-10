import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib
import cv2
import numpy as np

# load the img
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((100, 100)) 
    image = image.convert('L')  
    image_array = np.array(image)
    image_array_flat = image_array.flatten()
    return image_array_flat

# age and sleep predition here (I have catagorized the age section into a rage i.e 18-21 yrs , 22-26yrs rather individual age )
def predict_age_and_sleeping_status(image_path):

    age_model = joblib.load("age_prediction_model.pkl")
    sleep_model = joblib.load("sleep_detection_model.pkl")

    image_array_flat = preprocess_image(image_path)

    age_prediction = age_model.predict([image_array_flat])
    sleep_prediction = sleep_model.predict([image_array_flat])

    # age range
    age_range = f"{age_prediction[0]}-{age_prediction[0]+5}"  

    # Sleep
    sleeping_status = "Sleeping" if sleep_prediction == 1 else "Not Sleeping"

    return age_range, sleeping_status

# basic gui part 
def gui():
    file_path = filedialog.askopenfilename()
    if file_path:
        age_range, sleeping_status = predict_age_and_sleeping_status(file_path)
        result_label.config(text=f"Age Range: {age_range}\nSleeping Status: {sleeping_status}")

root = tk.Tk()
root.title("Age and Sleep Prediction")


select_button = tk.Button(root, text="Select Image", command=gui)
select_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
