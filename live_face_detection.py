import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

#loop throught every frame of the video 
while True:
    #getting the current frame from video as an image 
    ret, current_frame = video_capture.read()
    if not ret:
        break
    
    #resizing the current frame to make it 1/4 time samller to insrease the processing speed
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    
    faces_to_detect = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=3 ,model="hog")
    
    #looping through every face locations in the frame
    for index, current_face_location in enumerate(faces_to_detect):
        #splitting the tuple to get 4 positons values of the current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        #change the magnitude to the original size of the frame 
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        
        #printing the location of the current face
        #print(f'Found face {index+1} at top: {top_pos}, right: {right_pos}, bottom: {bottom_pos}, left: {left_pos}')
        
        #making a rectangle around the face 
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 255, 0), 2)
    
    #playing the current video stream 
    cv2.imshow("Webcam Video", current_frame)
    
    #stops the stream if the 'q' key is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#closses all the window
video_capture.release()
cv2.destroyAllWindows()
