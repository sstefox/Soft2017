import numpy as np
import cv2
import Person
import time


cap = cv2.VideoCapture('videos/video2.mp4') # opening files, right now video4

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)  # Create the background substractor
kernelOp = np.ones((3, 3), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

# Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1
areaTH = 120
areaMAX = 500

while (cap.isOpened()):
    ret, frame = cap.read()  # read a frame

    fgmask = fgbg.apply(frame)  # Use the substractor
    try:
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        # Opening (erode->dilate)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        # Closing (dilate -> erode)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

        # crtanje platforme PLATFORM boundaries
        line1 = np.array([[180, 110], [470, 90], [525, 450], [170, 460], [180, 110]], np.int32).reshape((-1, 1, 2))
        frame = cv2.polylines(frame, [line1], False, (255, 0, 0), thickness=2)

        line2 = np.array([[180, 110], [470, 90]], np.int32).reshape((-1, 1, 2))
        line3 = np.array([[525, 450], [170, 460]], np.int32).reshape((-1, 1, 2))

    except:
        # if there are no more frames to show...
        print('EOF')
        break

    _, contours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3, 8)
        area = cv2.contourArea(cnt)
        if (area > areaTH) & (area < areaMAX):
            #################
            #   TRACKING    #
            #################
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            new = True
            for i in persons:
                if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:

                    new = False
                    i.updateCoords(cx, cy)
                    break
            if new == True:
                p = Person.MyPerson(pid, cx, cy, max_p_age)
                persons.append(p)
                pid += 1
            ################
            #   DRAWING    #
            ################
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)


    cv2.imshow('Frame', frame)

    # Abort and exit with 'Q' or ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()  # release video file
cv2.destroyAllWindows()  # close all openCV windows