import face_recognition as fc
import cv2
import numpy as np
import pickle


class Person:
    def __init__(self, person, status=True):
        print(f"Processing {person}")
        self.name = person
        self.authorized = status
        self.image = fc.load_image_file("images/" + person + "/1.png")
        self.encoding = fc.face_encodings(self.image)[0]
        print(f"{person} processed")


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
]

print("\nCreating File")
f = open("people.pickle", "wb")
pickle.dump(People, f)
f.close()
print("Pickle created")

pickle_in = open("people.pickle", "rb")
a = pickle.load(pickle_in)
print(a[0].name)
