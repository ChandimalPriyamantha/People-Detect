# -*- coding: utf-8 -*-

import cv2
import playsound

print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('By Aditya Yogish Pai and Aditya Baliga B')

video_src = 'pedestrians.avi'
count = 1
cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
#cap.set(10,100)

bike_cascade = cv2.CascadeClassifier('faces.xml')


def play_audio():
    playsound.playsound("air_raid.wav",True)

while True:
    ret, img = cap.read()

    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bike = bike_cascade.detectMultiScale(gray, 1.3, 2)

    for (a, b, c, d) in bike:
        cv2.rectangle(img,(a, b),(a + c, b + d),(0, 255, 210),4)
    
    cv2.imshow('video', img)
   

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Image/Scanned_"+str(count)+".jpg",img)
        cv2.rectangle(img,(0, 200),(640,300),(0,255, 0),cv2.FILLED)
        cv2.putText(img,"Scan Saved", (150, 265),cv2.FONT_HERSHEY_DUPLEX,2,(0, 0, 255),2)
        cv2.imshow("result",img)
        cv2.waitKey(500)
        count +=1


cv2.destroyAllWindows()
