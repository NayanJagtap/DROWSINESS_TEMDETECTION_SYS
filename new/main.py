#this is opencv library for basic image processing functions
#to install opencv "c conda-forge opencv"
#or !pip install opencv-python
import cv2
#this is for array related functions
import numpy as np

#installing cmake
#!pip install cmake

#checking python version
#!python --version



#   installing dlib via file
# !pip install "C:\Users\NAYAN DINKAR JAGTAP\Downloads\Dlib-python whl packages\Dlib-python whl packages\dlib-19.22.99-cp38-cp38-win_amd64.whl"


#dlib is used for deeplearning based modules and face landmark detection
import dlib


#installing imutils
#!pip install imutils


from imutils import face_utils


#initializing the camera and taking the instance
cap=cv2.VideoCapture(0)

#intializing the face detector and landmark detector
detector=dlib.get_frontal_face_detector()
#get frontal face detector is an inbuilt function of dlib library which gives dlib's machine learning frontal face detector
#which is more accurate than harr cascasde of opencv,it do not requires any input as a file

#this will help in predicting 68 crutial land marks like eye-brows,lips,etc and it is commonly used in augumented reality like 
#snapchat filters
predictor=dlib.shape_predictor("C:\\Users\\NAYAN DINKAR JAGTAP\\Downloads\\archive\\shape_predictor_68_face_landmarks.dat")

#current states of variables
sleep=0
drowsy=0
active=0
status=""
color=(0,0,0)

#the distance between two points,lets say one point is at one end of right eye and other point is at one end of the left eye so 
#the distance between both the points can be calculated by eucledian distance,but here is the inbuit formula for that.
def compute(ptA,ptB):
    dist=np.linalg.norm(ptA-ptB)
    return dist

#one eye has 6 crutial points
#     .  .    
#  .        .
#     .  .
def blinked(a,b,c,d,e,f): 
    up=compute(b,d)+compute(c,e) #adding both short distances
    down=compute(a,f) #computing distance between the long distance 
    ratio=up/(2.0*down)
    
#this ratio will help us in determining whether the eye is closed,blinking or not etc 
#after experimenting 0.25 is the determined ratio or a threshold ratio  if greater than 0.25 then eye is opened else closed

#checking whether the eye is blonking or not
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0
    
while True:
    _,frame= cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray) #calling frontal face detector
    #detected face in faces array
    face_frame=frame.copy()
    for face in faces: #the rectangle generated will be because of this 
        x1=face.left()          #these are the coordinates for the rectangle
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        
        face_frame=frame.copy()
        cv2.rectangle(face_frame,(x1,y1),(x2,y2),(0,255,0),2)
        
        landmarks=predictor(gray,face)
        landmarks=face_utils.shape_to_np(landmarks) #shaped into numpyarray
        
        left_blink=blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
        right_blink=blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])
        
        #judging whether the driver is blinling,sleepy,active etc
        
        if(left_blink==0 or right_blink==0):   #in the blinked function there are 6 parameters of the eyes a,b,c,d,e,f,ratio
            #is calculated and 0,1,2 is returned according to it these three conditions will execute
            sleep+=1
            drowsy=0
            active=0
            if(sleep>6):   #if sleepy status exceeds 6 frames then
                status="!!!SLEEPING!!!"
                color=(255,0,0)


        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status="DROWSY !"
                color=(0,0,225)
                
                    
        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status="ACTIVE:"
                color=(0,255,0)

        cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)#text putting(conditions,status etc),colors putting

        for n in range(0,68):
            (x,y)=landmarks[n]
            cv2.circle(face_frame,(x,y),1,(255,0,0),-1)  #these are the landmarks 
    
    cv2.imshow("FRAME",frame)
    cv2.imshow("RESULT OF DETECTOR",face_frame)
    key=cv2.waitKey(1)
    if key==27:
        break
    
        
        
        

