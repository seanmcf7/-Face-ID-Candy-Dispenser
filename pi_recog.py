import face_recognition as fc
import cv2
import numpy as np
import pickle
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

class Person:
    def __init__(self, person, status=True):
        self.name = person
        self.authorized = status
        self.image = fc.load_image_file("images/" + person + "/1.png")
        self.encoding = fc.face_encodings(self.image)[0]


f = open("people.pickle", "rb")
People = pickle.load(f)
f.close()

# Camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 2
rawCapture = PiRGBArray(camera, size=(640, 480))

# Give camera time to warm up
time.sleep(0.1)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    '''
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    '''

    # Only process every other frame of video to save time
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        cv_image = frame.array
        image = cv_image[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = fc.face_locations(image)
        face_encodings = fc.face_encodings(image, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = fc.compare_faces(list(p.encoding for p in People), face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fc.face_distance(list(p.encoding for p in People), face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = People[best_match_index].name

            face_names.append(name)
        rawCapture.truncate(0)

    # process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            '''
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            '''

            # Draw a box around the face
            cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(cv_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(cv_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', cv_image)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
