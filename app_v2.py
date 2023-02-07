import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from pygame import mixer

frame_count = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
sleep = 0
drowsy = 0
active = 0
status=""

def detect(image_url):
    global frame_count, detector, predictor, sleep, drowsy, active, status
    
    def cal_yawn(landmarks):
        top_lip = landmarks[50:53]
        top_lip = np.concatenate((top_lip, landmarks[61:64]))
    
        low_lip = landmarks[56:59]
        low_lip = np.concatenate((low_lip, landmarks[65:68]))
        
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
    
        distance = dist.euclidean(top_mean,low_mean)
        return distance
    
    mixer.init()
    # sound= mixer.Sound(r'alarm.wav')


    yawn_thresh = 35

    def compute(ptA,ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(a,b,c,d,e,f):
        up = compute(b,d) + compute(c,e)
        down = compute(a,f)
        ratio = up/(2.0*down)

        if(ratio>0.25):
            return 2
        elif(ratio>0.21 and ratio<=0.25):
            return 1
        else:
            return 0
    
    frame = cv2.imread(image_url)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_frame = frame.copy()
    
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        lip_dist = cal_yawn(landmarks)
        if lip_dist > yawn_thresh:
            status = 'Sleeping'

        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            if(sleep>6):
                status = 'Sleeping'

        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status = 'Drowsy'
        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status = 'Awake'
            # else just return the previous status

        return status if status != '' else 'Detecting...'
    return 'No Person Found!'
                