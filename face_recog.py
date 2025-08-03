import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
import winsound

# To track names already marked today in this session
marked_names_today = set()
today_date = datetime.now().strftime('%Y-%m-%d')

# Path to folder containing known face images
path = 'known_faces'
images = []
classNames = []

# Load all known face images and extract names
for filename in os.listdir(path):
    img = cv2.imread(f'{path}/{filename}')
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])

# Encode all known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

encodeListKnown = findEncodings(images)
print('Face encoding completed.')

# Function to mark attendance
def markAttendance(name):
    global marked_names_today
    time_str = datetime.now().strftime('%H:%M:%S')
    date_str = datetime.now().strftime('%Y-%m-%d')

    if name in marked_names_today:
        return  # Already marked in this session

    try:
        df = pd.read_csv('Attendance.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Date', 'Name', 'Time'])

    # Check if already marked in CSV today
    if not ((df['Date'] == date_str) & (df['Name'] == name)).any():
        df.loc[len(df)] = {'Date': date_str, 'Name': name, 'Time': time_str}
        df.to_csv('Attendance.csv', index=False)
        winsound.MessageBeep()
        print(f'✅ Attendance marked for {name} at {time_str}')
    else:
        marked_time = df.loc[(df['Date'] == date_str) & (df['Name'] == name), 'Time'].values[0]
        print(f'ℹ️ Already marked earlier today: {name} at {marked_time}')

    marked_names_today.add(name)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame for faster processing
    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
