import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Used an image loding function from one of my older projects( Attempt 12)
def load_data(dataset_path):
    images = []
    labels = []
    for age_folder in os.listdir(dataset_path):
        age_folder_path = os.path.join(dataset_path, age_folder)
        if os.path.isdir(age_folder_path):
            for yawning_status in os.listdir(age_folder_path):
                yawning_status_path = os.path.join(age_folder_path, yawning_status)
                if os.path.isdir(yawning_status_path):
                    label = 1 if yawning_status == 'yawn' else 0
                    for image_file in os.listdir(yawning_status_path):
                        image_path = os.path.join(yawning_status_path, image_file)
                        image = cv2.imread(image_path)
                        image = cv2.resize(image, (100, 100)) 
                        images.append(image)
                        labels.append(int(age_folder.split('-')[0]))  
    return np.array(images), np.array(labels)

# dataset
dataset_path = "dataset"
images, labels = load_data(dataset_path)

# Split ->> training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# convert to gray
X_train_gray = np.mean(X_train, axis=3, keepdims=True)
X_test_gray = np.mean(X_test, axis=3, keepdims=True)

# Flatten 
X_train_flat = X_train_gray.reshape(X_train_gray.shape[0], -1)
X_test_flat = X_test_gray.reshape(X_test_gray.shape[0], -1)

#  used SVM here for age (attempt 4)
svm_age_classifier = SVC(kernel='linear', random_state=42)
svm_age_classifier.fit(X_train_flat, y_train)


y_pred_age = svm_age_classifier.predict(X_test_flat)

# Evaluate
accuracy_age = accuracy_score(y_test, y_pred_age)
print("Age Prediction Accuracy:", accuracy_age)

# model saving
joblib.dump(svm_age_classifier, "age_prediction_model.pkl")

# used SVM for sleep prediction 
svm_sleep_classifier = SVC(kernel='linear', random_state=42)
svm_sleep_classifier.fit(X_train_flat, y_train) 


y_pred_sleep = svm_sleep_classifier.predict(X_test_flat)


accuracy_sleep = accuracy_score(y_test, y_pred_sleep)
print("Sleep Detection Accuracy:", accuracy_sleep)

# Save
joblib.dump(svm_sleep_classifier, "sleep_detection_model.pkl")
