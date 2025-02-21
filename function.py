import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Mediapipe detection function
def mediapipe_detection(image, model):
    if image is None or image.size == 0:  # Check if the image is empty
        raise ValueError("Input image is empty or None. Check your input source.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False                  # Image is not writeable
    results = model.process(image)                 # Make predictions
    image.flags.writeable = True                   # Image is writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert RGB back to BGR
    return image, results

# Draw landmarks with style
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

# Extract keypoints from Mediapipe results
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
            return np.concatenate([rh])
    return np.zeros(21*3)  # Return zeros if no landmarks are found

# Configuration parameters
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
no_sequences = 200
sequence_length = 30
