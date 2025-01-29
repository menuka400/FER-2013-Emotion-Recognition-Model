import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("C:\\Users\\menuk\\Desktop\\New folder\\emotion_detection_model_100epochs.h5")

# Define emotion categories
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to preprocess the input image for prediction
def preprocess_input(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))             # Resize to 48x48 pixels
    image = image / 255.0                           # Normalize pixel values
    image = np.expand_dims(image, axis=0)           # Add batch dimension
    image = np.expand_dims(image, axis=-1)          # Add channel dimension
    return image

print("Press 'q' to exit.")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face using OpenCV's Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for emotion detection
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the ROI
        processed_roi = preprocess_input(face_roi)
        
        # Predict emotion
        predictions = model.predict(processed_roi)
        max_index = np.argmax(predictions[0])  # Get index of the highest probability
        emotion = emotion_labels[max_index]    # Map index to emotion label
        
        # Display emotion on the video feed
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the video feed
    cv2.imshow("Emotion Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
