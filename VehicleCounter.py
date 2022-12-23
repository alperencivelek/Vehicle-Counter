import cv2
import numpy as np
from time import sleep

length_min=80 #dikdortgen genisligi
height_min=80 #dikdrtgen yuksekligi
offset=6 #pixel başına hatalı izin
line_position=550 #cizginin konumu
delay=60 #fps

detect = []
vehicles=0

def fonksiyon (x,y,w,h):#aracın videodaki konumu
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detected_video=cv2.VideoCapture('video2.mp4')#okunacak video
subtractor=cv2.bgsegm.createBackgroundSubtractorMOG()#siyah beyaz pixel ayrıştırma

while True:
    ret, frame1 = detected_video.read()
    frequency = float(1/delay)
    sleep(frequency)
    toGray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    Blurred=cv2.GaussianBlur(toGray,(3,3),5)
    img_sub=subtractor.apply(Blurred)
    dilated=cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    widened = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    widened = cv2.morphologyEx(widened, cv2.MORPH_CLOSE, kernel)
    contour, h = cv2.findContours(widened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, line_position), (1200, line_position), (255, 127, 0), 3)
    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        valid_genislik = (w >= length_min) and (h >= height_min)
        if not valid_genislik:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = fonksiyon(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (line_position + offset) and y > (line_position - offset):
                vehicles += 1
                cv2.line(frame1, (25, line_position), (1200, line_position), (0, 127, 255), 3)
                detect.remove((x, y))
                print("Car Detected : " + str(vehicles))

    cv2.putText(frame1, "VEHICLE COUNT : " + str(vehicles), (450, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", widened)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
