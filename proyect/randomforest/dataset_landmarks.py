import polars as pl
import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle

# Set up hand detector
mp_hands = mp.solutions.hands  # Initialize MediaPipe Hands solution
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing hand landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Utility for drawing styles

# Initialize the Hands model with static image mode and minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=.3)

DATA_DIR = './data'  # Directory containing the dataset

data = []  # List to store the landmark data
labels = []  # List to store the corresponding labels

# Loop through each directory in the dataset
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store landmarks for the current image
        x_ = []  # List to store x-coordinates of landmarks
        y_ = []  # List to store y-coordinates of landmarks

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Read the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB

        results = hands.process(img_rgb)  # Process the image to detect hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks: this is just a preview
                '''
                mp_drawing.draw_landmarks(img_rgb, 
                                        hand_landmarks, 
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                plt.figure()
                plt.imshow(img_rgb)
                '''
                # Loop through each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Get x-coordinate of the landmark
                    y = hand_landmarks.landmark[i].y  # Get y-coordinate of the landmark

                    x_.append(x)  # Append x-coordinate to the list
                    y_.append(y)  # Append y-coordinate to the list

                    # Normalize the coordinates by subtracting the minimum value
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)  # Append the normalized landmarks to the data list
            labels.append(dir_)  # Append the label to the labels list

# Save the data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
