def distance(p1,p2):
    return np.sum((p2 - p1)**2)**.5

def knn(X,Y,test,k=5):
    m = X.shape[0]
    
    d = []
    for i  in range(m):
        dist = distance(test,X[i])
        d.append((dist,Y[i]))
    
    d = np.array(sorted(d))[:,1]
    d = d[:k]
    t =  np.unique(d,return_counts=True)
    idx = np.argmax(t[1])
    pred = int(t[0][idx])
        
    return pred


import numpy as np
import os 
import cv2


###### KNN #######

label = []
class_id = 0
names = {}
face_data = []

#load data 
data_path = "../../sampling/face_data/"
for fx in os.listdir(data_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item =np.load(data_path+fx)
        face_data.append(data_item)
        
        #label
        target = class_id*np.ones((data_item.shape[0],))
#         print(target.shape)
        label.append(target)
        class_id+=1
        
 #concatenate  
face_label = np.concatenate(label,axis=0)
face_X = np.concatenate(face_data,axis=0) 
        
#reshape label
face_Y = face_label.reshape((-1,1))
        
#concatenate X and Y together 
training_set = np.concatenate((face_X,face_Y),axis=1)
        
        
        

# read from videocapture

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    
    #rectangular frame
    for face in faces:
        x,y,w,h = face
        #cv2.rectangle(face,(x,y),(x+w,y+h),(255,255,0),1) not here but after prediction
        offset =10
        face_sec = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_sec = cv2.resize(face_sec,(100,100))
        
        out = knn(training_set[:,:-1],training_set[:,-1],face_sec.flatten())
        
        #displeay text on screen
        pred_name = names[int(out)]
        cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
        #display rectangle
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(255,255,0),1)
        
        
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    cv2.imshow("face",gray_frame)




