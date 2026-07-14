import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import imutils
print("✅ All libraries loaded successfully!")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam not detected!")
        break
    cv2.imshow("Webcam Test - Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam test complete!")