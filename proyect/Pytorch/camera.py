import cv2
import numpy as np
import onnxruntime as ort

# Function to center crop the frame
def center_crop(frame):
    h, w, _ = frame.shape  # Get the height and width of the frame
    start = abs(h - w) // 2  # Calculate the starting point for cropping
    if h > w:
        return frame[start: start + w]  # Crop the height if it's greater than the width
    return frame[:, start: start + h]  # Crop the width if it's greater than the height

def main():
    # Constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')  # List of letters corresponding to the model's output
    mean = 0.485 * 255.  # Mean value for normalization
    std = 0.229 * 255.  # Standard deviation for normalization

    # Create a runnable session with the exported ONNX model
    ort_session = ort.InferenceSession("signlanguage.onnx")

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if the frame is not captured

        # Preprocess the frame
        frame = center_crop(frame)  # Center crop the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert the frame to grayscale
        x = cv2.resize(frame, (28, 28))  # Resize the frame to 28x28 pixels
        x = (x - mean) / std  # Normalize the frame

        # Reshape the frame to match the model's input shape
        x = x.reshape(1, 1, 28, 28).astype(np.float32)

        # Run the model to get the prediction
        y = ort_session.run(None, {'input': x})[0]

        # Get the predicted letter
        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        # Display the predicted letter on the frame
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()