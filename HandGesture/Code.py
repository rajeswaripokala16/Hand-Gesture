import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

# Set threshold for pinch gesture
THRESHOLD = 0.05  # Adjust this value according to your need
key_to_press = 'space'
pressed = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get coordinates for thumb tip and index tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Draw circles on tips
            h, w, c = frame.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.circle(frame, thumb_pos, 8, (0, 255, 0), -1)
            cv2.circle(frame, index_pos, 8, (255, 0, 0), -1)
            
            # Calculate distance between points
            distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
            cv2.putText(frame, f'Distance: {distance:.3f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            # Key press logic
            if distance < THRESHOLD and not pressed:
                pyautogui.press(key_to_press)
                pressed = True
            elif distance >= THRESHOLD:
                pressed = False
            
    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
