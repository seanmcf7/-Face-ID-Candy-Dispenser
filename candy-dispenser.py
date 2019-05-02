import face_recognition as fc
import cv2
import numpy as np
from gpiozero import Servo, Button, DigitalOutputDevice


def check_if_allowed(people):
    n = True
    for i in people:
        if not i.authorized:
            n = False
    if n:
        print("Enjoy some candy")
        # Dump candy
    else:
        print("Access Denied")


class Person:
    def __init__(self, person, status=True):
        self.name = person
        if person == "Unknown":
            self.authorized = False
            self.image = None
            self.encoding = None
        else:
            self.authorized = status
            self.image = fc.load_image_file("images/" + person + "/1.png")
            self.encoding = fc.face_encodings(self.image)[0]


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

People = [
    Person("Mom"),
    Person("Dad"),
    Person("Shane", False),
    Person("Alyssa"),
    Person("Megan", False),
    Person("Collin"),
    Person("Erin"),
    Person("Sean"),
    Person("Marissa", False),
    Person("Ryan"),
    Person("Unknown")
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Servo, button, and LCD details
corrections = 0.00
maxPW = (2.0 + corrections) / 1000
minPW = (1.0 - corrections) / 1000
dispenser = Servo(17, min_pulse_width=minPW, max_pulse_width=maxPW)
button = Button(4)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = fc.face_locations(rgb_small_frame)
        face_encodings = fc.face_encodings(rgb_small_frame, face_locations)

        people_in_frame = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = fc.compare_faces(list(p.encoding for p in People), face_encoding)
            name = People[-1]

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fc.face_distance(list(p.encoding for p in People), face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = People[best_match_index]

            people_in_frame.append(name)

            # Check if button is pressed and all users are approved
            button.when_deactivated(check_if_allowed(people_in_frame))

    process_this_frame = not process_this_frame

    # Check if user is shutting down
    if button.active_time >= 6:
        print("Shutting down")
        break
    elif button.active_time >= 5:
        print("Shutting down in 1")
    elif button.active_time >= 4:
        print("Shutting down in 2")
    elif button.active_time >= 3:
        print("Shutting down in 3")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
print("Goodbye")
# Close servo
# Turn LCD screen off
