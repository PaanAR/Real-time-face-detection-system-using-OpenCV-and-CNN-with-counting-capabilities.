#STTHK3013 - Pattern Recognition and Analysis
#Base code - Face Recognition using a Convolutional Neural Networks & Transfer Learning Model
#Assignment #4 (individual)

#important things to do:
#1 - make sure you copy haarcascade_frontalface_default.xml file and paste at the same folder of your python code
#2 - install OpenCV library (using PIP > pip install opencv-python) or Conda > conda install -c conda-forge opencv
#3 - install tensorflow library 
#4 - press key Q to exit

#import all libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Build the CNN model
def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Added new layers
        Conv2D(128, (3, 3), activation='relu'),  # New convolutional layer
        MaxPooling2D(pool_size=(2, 2)),         # New pooling layer
        tf.keras.layers.Dropout(0.25),          # New dropout layer for regularization
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# Initialize CNN (pretend it's pre-trained for this example)
cnn_model = build_cnn()

# Load weights from a pre-trained model (if available)
# cnn_model.load_weights("pretrained_face_model.h5")

# Step 2: Setup OpenCV for real-time face detection
# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess face for CNN input
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))  # Resize to model input size
    face_img = face_img / 255.0  # Normalize pixel values
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    return face_img

# Step 3: Real-time video stream and face detection
cap = cv2.VideoCapture(0)  # Open webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess face for CNN model
        processed_face = preprocess_face(face_roi)

        # Predict using CNN
        prediction = cnn_model.predict(processed_face)
        label = "Face" if prediction[0] > 0.5 else "No Face"

        # Draw bounding box and label
        color = (0, 255, 0) if label == "Face" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Count and display number of faces
    face_count = len(faces)
    cv2.putText(frame, f'Faces detected: {face_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Face Detection', frame)

    # press key q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
