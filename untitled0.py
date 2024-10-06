# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:50:45 2024

@author: yaten
"""

import cv2
import face_recognition

# Load the original image
original_image = cv2.imread('images/test/group_photo.jpg')

# Load the test image
test_image = face_recognition.load_image_file('images/test/group_photo.jpg')

# Detect faces in the test image
faces_to_detect = face_recognition.face_locations(test_image, model="hog")

# Iterate through each face found in the test image
for (top, right, bottom, left) in faces_to_detect:
    # Draw a rectangle around the face
    cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)
    
# Display the result
cv2.imshow("faces identify", original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
