import cv2
import numpy as np

cap = cv2.VideoCapture('HallWayTracking/videos/002.avi');
ret, frame1 = cap.read()
ret, frame2 = cap.read()
success, image = cap.read()

data_homo = np.loadtxt('HallWayTracking/homography/002.txt', delimiter=',', dtype=float)

tl = (425, 13)
bl = (43, 473)
tr = (476, 13)
br = (550, 473)
x1 = np.float32([tl, bl, tr, br])
x2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

while success:
    success, image = cap.read()
    frame1 = cv2.resize(image, (640, 480))
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations = 3)
    сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for contour in сontours:
        (x, y, w, h) = cv2.boundingRect(contour) 
        if cv2.contourArea(contour) > 1000: 
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame1, tl, 5, (0,0,255), -1)
            cv2.circle(frame1, bl, 5, (0,0,255), -1)
            cv2.circle(frame1, tr, 5, (0,0,255), -1)
            cv2.circle(frame1, br, 5, (0,0,255), -1)

            matrix = cv2.getPerspectiveTransform(x1, x2)
            bird_frame = cv2.warpPerspective(frame1, matrix, (640, 480))

            cv2.imshow('Frame', frame1)
            cv2.imshow('Eye-bird show', bird_frame)        
            x_obj = 1
            y_obj = 1

    frame1 = frame2 
    ret, frame2 = cap.read()   
    if cv2.waitKey(15) == 42:
        break

cap.release()
cv2.destroyAllWindows()
