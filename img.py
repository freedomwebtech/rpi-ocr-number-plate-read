import cv2
import time
cpt = 0
maxFrames = 1200 # if you want 5 frames only.

cap=cv2.VideoCapture('tc.mov')
while True:
#while cpt < maxFrames:
    ret, frame = cap.read()
    
    
    if not ret:
        break
    frame=cv2.resize(frame,(1020,500))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite("/home/pi/Downloads/paddleocrnumberplateread-main/images/img_%d.jpg" %cpt, frame)
    time.sleep(0.01)
    cpt += 1
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
