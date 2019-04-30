import numpy as np
import cv2
import pickle

# This cascade is used to detect faces, but does not identify them
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# Open trained data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Retrieve labels from trained data
og_labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
        # print(f"x:{x} y:{y} w:{w} h:{h}")
        roi_gray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 50:
            name = labels[id_]
            cv2.putText(frame, name, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)

        img_item = "my-face.png"
        cv2.imwrite(img_item, roi_gray)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
