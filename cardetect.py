import numpy as np
import cv2
import datetime
import os



car_cascade = cv2.CascadeClassifier("cars.xml")

#load the data
with np.load('knn_data.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']
    
# clase 0 = sedan, clase 1 = bus

cap = cv2.VideoCapture("traffic4.mp4")
kernel = np.ones((3,3),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'XVID') esta DEPRECATED en phyton3
## ahora sustituimos por:
fourcc = cv2.VideoWriter_fourcc(*'mpeg')
cn = 0
flag = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        car = car_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(80, 80), maxSize=(500,500))
        for (x,y,w,h) in car:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.putText(frame,'car',(x+w+10,y+h), font, 0.5, (100,255,0), 1, cv2.CV_AA)
            #cv2.putText(frame,str(cn),(x+w+10,y+h+10), font, 0.5, (100,255,0), 1, cv2.CV_AA)
            cv2.putText(frame,'car',(x+w+10,y+h), font, 0.5, (100,255,0), 1, cv2.LINE_AA)
            cv2.putText(frame,str(cn),(x+w+10,y+h+10), font, 0.5, (100,255,0), 1, cv2.LINE_AA)
            roi_color = frame[y:y+h, x:x+w]
            
            
        #if np.any(car) != False:        
            r,h,c,w = y,h,x,w  # simply hardcoded the values
            track_window = (x,y,w,h)
            
            # set up the ROI for tracking
            roi = frame[r:r+h, c:c+w]
            hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

            # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            flag = 1
            
            roi_color = cv2.resize(roi_color, (100,100))
            

        #if flag == 1:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            
            # apply meanshift to get the new location
            #ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # Draw it on image
            #pts = cv2.cv.BoxPoints(ret)
            #pts = np.int0(pts)
            #cv2.polylines(frame,[pts],True, 255,2)
            roi_gray = cv2.cvtColor( roi_color, cv2.COLOR_BGR2GRAY)
            
            test = roi_gray.reshape(-1,10000).astype(np.float32)
        
            # Initiate kNN, train the data, then test it with test data for k=1
            #deprecated
            #knn = cv2.KNearest()
            knn = cv2.ml.KNearest_create()
            knn.train(train, train_labels)
            ret,result,neighbours,dist = knn.find_nearest(test,k=7)
            
            
            
            #cv2.putText(frame,'asd',(x+w+10,y+h-10), font, 0.5, (0,100,80), 1, cv2.CV_AA)
            print("____________________")
            print("car:",cn)
            print("result:",result)
            print("neighbours", neighbours)
            print("____________________")
            
            if result == 0:
                cv2.putText(frame,'Sedan',(x+w+10,y+h-20), font, 0.5, (0,255,0), 1, cv2.CV_AA)
                cv2.putText(roi_color,'Sedan',(30,90), font, 0.5, (0,255,0), 1, cv2.CV_AA)
                roi_color = cv2.resize(roi_color, (200,200))
                cv2.imshow('Sedan',roi_color)
            if result == 1:
                cv2.putText(frame,'Autobus',(x+w+10,y+h-20), font, 0.5, (0,255,0), 1, cv2.CV_AA)
                cv2.putText(roi_color,'Autobus',(30,90), font, 0.5, (0,255,0), 1, cv2.CV_AA)
                roi_color = cv2.resize(roi_color, (200,200))
                cv2.imshow('Autobus',roi_color)
                
            cn = cn+1
            
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
            
cap.release()
cv2.destroyAllWindows()