import numpy as np
import cv2

cap = cv2.VideoCapture(0)
num_down=2
num_bilateral=4


while(True):
    
    ret, frame = cap.read()

    print(frame.shape)
    #resizing for optimal results
    frame=cv2.resize(frame,(1368,800))

    #flip to mirror
    frame=np.flip(frame,1)
    
    #downsampling using gaussian pyramid
    frame1=frame
    for _ in range(num_down):
        frame1=cv2.pyrDown(frame1)

    #repeatedly applying small bilateral filter
    for _ in range(num_bilateral):
        frame1=cv2.bilateralFilter(frame1,d=9,sigmaColor=9,sigmaSpace=7)

    #upsampling to original image
    for _ in range(num_down):
        frame1=cv2.pyrUp(frame1)

    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    blur=cv2.medianBlur(gray,7)
    edge=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2)

    #converting back to color image and bit-AND with color image
    edge=cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB)
    cartoon=cv2.bitwise_and(frame1,edge)
    stack=np.hstack([frame,cartoon])
    
    #display
    cv2.imshow('frame',cartoon)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()