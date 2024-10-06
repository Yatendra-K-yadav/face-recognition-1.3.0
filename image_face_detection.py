
import face_recognition
import cv2

#loading the image ti=o be detected 
image_for_detect = cv2.imread('images/test/group_photo.jpg')
#if image_for_detect is None:
    #print("Error: Could not read the image.")
#else:
    #winname = "test" 
    #cv2.imshow(winname, image_for_detect)
    # Wait for a key press to close the window, without this the image would appear and close instantly
    #cv2.waitKey(0) 
    # Close all OpenCV windows after pressing 0
    #cv2.destroyAllWindows() 

faces_to_detect = face_recognition.face_locations(image_for_detect,model="hog")

#printing the no. of faces found in the imgae
print('there are {} no of faces in this iamge '.format(len(faces_to_detect)))


if image_for_detect is None:
    print("Error: Could not read the image.")
else:    
    #looping throught all the face location and slicing the no of faces found in the image
    for index, current_face_location in enumerate(faces_to_detect):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        current_face_iamge = image_for_detect[top_pos:bottom_pos,left_pos:right_pos]
        cv2.imshow("face no "+str(index+1),current_face_iamge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

















