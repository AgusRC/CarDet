import numpy as np
import cv2
import datetime
import os


def whcar(frame,gray):
    car = car_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50), maxSize=(300,300))
    for (x,y,w,h) in car:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,100,80),2)
            #cv2.putText(frame,'car',(x+w+10,y+h), font, 0.5, (0,100,80), 1, cv2.CV_AA)
            roi_color = frame[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (100,100))
            tiempo = datetime.datetime.now()
            print tiempo
            dia = "cardetect-"+str(tiempo.year)+"-"+str(tiempo.month)+"-"+str(tiempo.day)
            if(os.path.exists(dia) == False):
                os.mkdir(dia)
            strtiempo = str(tiempo.hour)+"-"+str(tiempo.minute)+"-"+str(tiempo.second)+".jpg"
            #print strtiempo
            cv2.imwrite(dia+"/car"+strtiempo,roi_color)                      #########Guarda los autos encontrados







cap = cv2.VideoCapture("vids/traffic2.mp4")
kernel = np.ones((3,3),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')

car_cascade = cv2.CascadeClassifier("cascades/cars.xml")


while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		#frame = cv2.flip(frame,1)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#gray = cv2.medianBlur(gray, 5)
		#gray = cv2.equalizeHist(gray)
		grayflip = cv2.flip(gray,1)
		gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
		#laplacian = cv2.Laplacian(gray,cv2.CV_64F)
		
		##Obtencion del tamao del frame
		framelargo = frame.shape[1]
		framealto = frame.shape[0]
                ##Linea divisora
        	#cv2.line(frame,(framelargo/2,0),(framelargo/2,framealto),(255,0,0),2)
        	
        	
                whcar(frame,gray)
		
		cv2.imshow('frame',frame)
                #cv2.imshow('grises',gray)
                #cv2.imshow('Lapace',laplacian)
    	
    	
    	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
        

cap.release()
cv2.destroyAllWindows()
