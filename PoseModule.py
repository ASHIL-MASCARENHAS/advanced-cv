"""
PoseModule.py
Includes AI Trainer with Data Logging for Data Science analytics.
"""
import cv2
import mediapipe as mp
import math
import csv
import datetime
import os

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.curl_count = 0
        self.dir = 0
        
        # Data Science: Initialize Data Log
        self.log_file = 'workout_data.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Angle', 'Count', 'Stage'])

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        lmList = self.findPosition(img, draw=False)
        if len(lmList) != 0:
            x1, y1 = lmList[p1][1:]
            x2, y2 = lmList[p2][1:]
            x3, y3 = lmList[p3][1:]

            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            if angle < 0: angle += 360
            
            # Simple logic to handle simple arm curls
            if angle > 180: angle = 360 - angle
            
            # Logic for counting
            stage = "Middle"
            if angle > 160:
                self.dir = 0 # Down
                stage = "Down"
            if angle < 30 and self.dir == 0:
                self.dir = 1 # Up
                self.curl_count += 0.5
                stage = "Up"
            
            # Log Data
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now(), int(angle), int(self.curl_count), stage])

            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
                cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            return angle

def run_trainer():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            # Right Arm
            detector.findAngle(img, 12, 14, 16)
        
        cv2.putText(img, f'Curls: {int(detector.curl_count)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("AI Trainer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            return 'quit'
    return 'back'

if __name__ == "__main__":
    run_trainer()