import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# 1. Load Calorie Data
calorie_data = pd.read_csv("d:/SHUN/internship/task5/food_calories.csv")  # CSV with Pizza, Waffles, and Soup

# 2. Load and Preprocess Image Dataset
def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for food in os.listdir(folder):
        food_folder = os.path.join(folder, food)
        if os.path.isdir(food_folder):
            label_map[label_counter] = food  # Mapping label to food name
            for filename in os.listdir(food_folder):
                img_path = os.path.join(food_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    labels.append(label_counter)
            label_counter += 1
    return np.array(images), np.array(labels), label_map

# 3. Create a CNN Model
def create_cnn_model(image_size=(64, 64, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=image_size))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # Output layer for 3 food categories
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Train the Model
def train_model(model, images, labels, image_size=(64, 64)):
    images = images / 255.0  # Normalize images
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Image augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Train the model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))
    return model

def predict_uploaded_image(image_path, model, label_map, image_size=(64, 64)):
    # Load and preprocess the uploaded image
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize image
        
        # Predict using the trained model
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)
        food_item = label_map[predicted_label]

        # Print predicted food item for debugging
        print(f"Predicted food item: {food_item}")
        
        # Check if the food item exists in the calorie data
        if food_item in calorie_data['Food'].values:
            calories = calorie_data.loc[calorie_data['Food'] == food_item, 'Calories'].values[0]
        else:
            print(f"Calories information not found for {food_item}")
            calories = "Unknown"

        # Display the image and prediction
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title(f"Food: {food_item}\nCalories: {calories} kcal")
        plt.axis('off')
        plt.show()
    else:
        print("Error: Unable to load the image.")

# 6. Main Function
def main():
    # Folder where food images are stored
    FOOD_FOLDER = "D:/SHUN/internship/task5/food_images"  # Replace with your folder path
    images, labels, label_map = load_images_from_folder(FOOD_FOLDER, image_size=(64, 64))

    # Create and train the CNN model
    model = create_cnn_model(image_size=(64, 64, 3))
    model = train_model(model, images, labels, image_size=(64, 64))

    # Save the trained model for future use
    model.save('food_calorie_estimator_model.h5')
    print("Model trained and saved.")

    # Upload and predict image
    print("\nUpload an image for prediction:")
    root = tk.Tk()
    root.withdraw()  # Hide tkinter root window
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if image_path:
        predict_uploaded_image(image_path, model, label_map, image_size=(64, 64))
    else:
        print("No file selected!")

if __name__ == "__main__":
    main()
