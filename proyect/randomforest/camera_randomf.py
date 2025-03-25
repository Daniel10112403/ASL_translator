import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the pre-trained model, scaler, and max_length from a pickle file
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']
max_length = model_dict['max_length']

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Set up MediaPipe hand detector and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.3)

# Dictionary to map model predictions to sign language characters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

while True:
    data_aux = []  # Auxiliary data for storing hand landmarks
    x_ = []  # List to store x-coordinates of hand landmarks
    y_ = []  # List to store y-coordinates of hand landmarks

    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        break  # Exit the loop if the frame is not captured successfully

    H, W, _ = frame.shape  # Get the height and width of the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
    results = hands.process(frame_rgb)  # Process the frame to detect hand landmarks
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            # Extract x and y coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize and append the coordinates to data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux = np.append(data_aux, x - min(x_))
                data_aux = np.append(data_aux, y - min(y_))

            # Ensure the length of data_aux is less than or equal to max_length
            if len(data_aux) <= max_length:
                # Pad sequences to the same length
                data_aux = np.pad(data_aux, (0, max_length - len(data_aux)), 'constant')

                # Standardize the data
                data_aux = scaler.transform([data_aux])

                # Calculate bounding box coordinates
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Predict the sign language character
                prediction = model.predict(data_aux)
                predicted_character = labels_dict[int(prediction[0])]

                # Draw the bounding box and predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Translator', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows