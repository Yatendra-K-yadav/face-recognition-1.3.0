# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 23:03:01 2024

@author: yaten
"""

import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

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

# Loop through every frame of the video
while True:
    # Getting the current frame from the video as an image
    ret, current_frame = video_capture.read()
    if not ret:
        break
    
    # Resizing the current frame to make it 1/4 times smaller to increase processing speed
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    
    faces_to_detect = face_recognition.face_locations(current_frame_small, model="hog")
    faces_to_encode = face_recognition.face_encodings(current_frame_small, faces_to_detect)

    # Iterate through each face found in the current frame
    for (top, right, bottom, left), face_encoding in zip(faces_to_detect, faces_to_encode):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name_of_person = "unknown face"
        
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # If a match is found in known_face_encodings, use the first match
        if True in matches:
            first_match_index = matches.index(True)
            name_of_person = known_face_names[first_match_index]
            
        # Draw a rectangle around the face
        cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left, bottom + 20), font, 0.5, (255, 255, 255), 1)
        
    # Display the result
    cv2.imshow("faces identify", current_frame)

    # Stops the stream if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closes all the windows
video_capture.release()
cv2.destroyAllWindows()
