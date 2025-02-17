import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, BackgroundTasks
import threading

app = FastAPI()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for exercise counting
counter_bicep = 0
counter_squats = 0
counter_lateral_raises = 0
stage_bicep = None
stage_squats = None
stage_lateral_raises = None

# Calculate angle function (as you already have)
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Exercise counting logic for Bicep Curls
def bicep_curl_exercise():
    cap = cv2.VideoCapture(1)  # Use the first camera
    global counter_bicep, stage_bicep
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angle
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage_bicep = "down"
                if angle < 30 and stage_bicep == 'down':
                    stage_bicep = "up"
                    counter_bicep += 1
                    print("Bicep Curl Counter:", counter_bicep)

            except:
                pass

            cv2.imshow('Bicep Curl Exercise', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()




    # Exercise counting logic for squats

def squats_exercise():
    cap = cv2.VideoCapture(1)  # Use the first camera
    global counter_squats, stage_squats
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angle
            try:
                landmarks = results.pose_landmarks.landmark
            
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)

                if angle > 160:
                    stage_squats = "down"
                if angle < 75 and stage_squats == 'down':
                    stage_squats = "up"
                    counter_squats += 1
                    print("Squats Counter:", counter_squats)

            except:
                pass

            cv2.imshow('Squats Exercise', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()



    # Exercise counting logic for lateral raises
def lateral_raises_exercise():
    cap = cv2.VideoCapture(1)  # Use the first camera
    global counter_lateral_raises, stage_lateral_raises
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angle
            try:
                landmarks = results.pose_landmarks.landmark
            
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
                # Calculate angle
                angle = calculate_angle(hip, shoulder, elbow)
                angle2 = calculate_angle(shoulder, elbow, wrist)

                # Curl counter logic
                if angle2 < 130:
                    print('Bad form')
                    bad_form = True  # Flag to track bad form
                else:
                    bad_form = False  # Reset if form is correct

                if angle < 20:
                    stage_lateral_raises = "down"

                if angle > 85 and stage_lateral_raises == 'down' and not bad_form:  # Only count if form is correct
                    stage_lateral_raises = "up"
                    counter_lateral_raises += 1
                    print("Good rep:", counter_lateral_raises) 

            except:
                pass

            cv2.imshow('Lateral raises Exercise', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Similar functions for squats and lateral raises...

# Route for Bicep Curl Exercise
@app.get("/bicep_curl")
async def bicep_curl(background_tasks: BackgroundTasks):
    background_tasks.add_task(bicep_curl_exercise)
    return {"message": "Bicep curl exercise started!"}

# Route for Squat Exercise
@app.get("/squats")
async def squats(background_tasks: BackgroundTasks):
    background_tasks.add_task(squats_exercise)  # Define squats_exercise similar to bicep_curl_exercise
    return {"message": "Squat exercise started!"}

# Route for Lateral Raises Exercise
@app.get("/lateral_raises")
async def lateral_raises(background_tasks: BackgroundTasks):
    background_tasks.add_task(lateral_raises_exercise)  # Define lateral_raises_exercise similar to bicep_curl_exercise
    return {"message": "Lateral raises exercise started!"}
