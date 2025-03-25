import os
import cv2

# Directory to save the collected images
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Create the directory if it doesn't exist

# Number of classes and dataset size
number_of_classes = 24  # Total number of classes
dataset_size = 100  # Number of images to collect per class

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Loop through each class
for j in range(number_of_classes):
    
    # Create a directory for the current class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Display a message to get ready
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Display the frame
        if cv2.waitKey(25) == ord('q'):  # Wait for the user to press 'q'
            break

    # Collect images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the webcam
        cv2.imshow('frame', frame)  # Display the frame
        cv2.waitKey(25)  # Wait for 25 milliseconds
        # Save the captured frame to the class directory
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1  # Increment the counter

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
