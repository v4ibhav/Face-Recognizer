#data creator
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0
face_data = []
data_path = "../../sampling/face_data/" 
person_name = input("enter your name: ")
while True:
    ret,frame =cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
   
    for face in faces[-1:]:
    	x,y,w,h = face
    	cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(255,0,255),1)
    	faces = sorted(faces,key = lambda f:f[2]*f[3])
    	offset = 10
    	face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
    	face_section = cv2.resize(face_section,(100,100))
    	skip +=1
    	if skip%10==0:
    		face_data.append(face_section)
    		_col = len(face_data)
	    	print(len(face_data))
    
    
    
    cv2.imshow("frame",gray_frame)
    cv2.imshow("face sec",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(data_path+person_name+".npy",face_data)
print("data succesfully saved at "+data_path+person_name+".npy")
cap.release()
cv2.destroyAllWindows()