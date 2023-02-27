import cv2
import numpy as np

cap = cv2.VideoCapture('HallWayTracking/videos/002.avi')
success, image = cap.read()

while success:
    success, image = cap.read()
    frame = cv2.resize(image, (640, 480))

    tl = (425, 13)
    bl = (43, 473)
    tr = (476, 13)
    br = (550, 473)

    cv2.circle(frame, tl, 3, (0,0,255), -1)
    cv2.circle(frame, bl, 3, (0,0,255), -1)
    cv2.circle(frame, tr, 3, (0,0,255), -1)
    cv2.circle(frame, br, 3, (0,0,255), -1)

    x1 = np.float32([tl, bl, tr, br])
    x2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

    matrix = cv2.getPerspectiveTransform(x1, x2)
    bird_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    cv2.imshow('Frame', frame)
    cv2.imshow('Eye-bird show', bird_frame)
    
    if cv2.waitKey(1) == 42:
        break
    
