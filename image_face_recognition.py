# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:50:45 2024

@author: yaten
"""

import cv2
import face_recognition

# Load the original image
original_image = cv2.imread('images/test/group_photo_2.0.jpg')

# Loading the sample images and encoding the faces
yatendra_image = face_recognition.load_image_file('images/sample/yatendra.jpg')
yatendra_face_encoding = face_recognition.face_encodings(yatendra_image)[0]

smit_image = face_recognition.load_image_file('images/sample/smit.jpg')
smit_face_encoding = face_recognition.face_encodings(smit_image)[0]

piyush_image = face_recognition.load_image_file('images/sample/piyush.jpg')
piyush_face_encoding = face_recognition.face_encodings(piyush_image)[0]

bhavya_image = face_recognition.load_image_file('images/sample/bhavya.jpg')
bhavya_face_encoding = face_recognition.face_encodings(bhavya_image)[0]

prithivraj_image = face_recognition.load_image_file('images/sample/prithivraj.jpg')
prithivraj_face_encoding = face_recognition.face_encodings(prithivraj_image)[0]

known_face_encodings = [yatendra_face_encoding, smit_face_encoding, piyush_face_encoding, bhavya_face_encoding, prithivraj_face_encoding]
known_face_names = ["yatendra", "smit", "piyush", "bhavya", "prithivraj"]           

# Load the test image
test_image = face_recognition.load_image_file('images/test/group_photo_2.0.jpg')

# Detect faces in the test image
faces_to_detect = face_recognition.face_locations(test_image, model="hog")
faces_to_encode = face_recognition.face_encodings(test_image, faces_to_detect)

# Iterate through each face found in the test image
for (top, right, bottom, left), face_encoding in zip(faces_to_detect, faces_to_encode):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name_of_person = "unknown face"

    # If a match is found in known_face_encodings, use the first match
    if True in matches:
        first_match_index = matches.index(True)
        name_of_person = known_face_names[first_match_index]
        
    # Draw a rectangle around the face
    cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Draw a label with a name below the face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left, bottom + 20), font, 0.5, (255, 255, 255), 1)
    
# Display the result
cv2.imshow("faces identify", original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
