import numpy as np
import cv2
import Person


f = open("out.txt", "w")
f.write("RA241/2013, Stefan Stankovic" + "\n")
f.write("file,count" + "\n")



for videoSnimak in range (1,11):
    videofajl = 'videos/video' + format(videoSnimak) + '.mp4'
    videoNaziv = 'video' + format(videoSnimak)

    cap = cv2.VideoCapture(videofajl) # otvramo video fajl

    w = cap.get(3)
    h = cap.get(4)

    line2 = np.array([[180, 110], [470, 90]], np.int32).reshape((-1, 1, 2))
    line3 = np.array([[525, 450], [170, 460]], np.int32).reshape((-1, 1, 2))

    upLimit = 110
    downLimit = 450

    pt1 = [180, upLimit];
    pt2 = [470, 90];
    pts_L1 = np.array([pt1, pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))
    pt3 = [525, downLimit];
    pt4 = [170, 460];
    pts_L2 = np.array([pt3, pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1, 1, 2))

    # Kreiramo background substractor
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    kernelOpt = np.ones((3, 3), np.uint8)
    kernelCle = np.ones((11, 11), np.uint8)

    # Variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    persons = []
    maxPersonAge = 5
    personID = 1
    # minimalna povrsina koju smatramo osobom
    areaTHR = 230
    # maksimalna porvrsina koju smatramo osobom
    areaMAX = 500

    while (cap.isOpened()):
        ret, frame = cap.read()  # ocitavamo frejm

        for i in persons:
            i.age_one()  # povecamo broj godina za ovaj frejm
        #########################
        #   PREDPROCESIRANJE    #
        #########################

        fgmaska = fgbg.apply(frame)  # Use the substractor
        fgmaska2 = fgbg.apply(frame)

        # Pokusavamo ocistiti frejm kako bi se mogle prepoznati osobe
        try:
            ret, imgBin = cv2.threshold(fgmaska, 200, 255, cv2.THRESH_BINARY)
            ret, imgBin2 = cv2.threshold(fgmaska2, 200, 255, cv2.THRESH_BINARY)
            # Opening (erode->dilate)
            maska = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernelOpt)
            maska2 = cv2.morphologyEx(imgBin2, cv2.MORPH_OPEN, kernelOpt)
            # Closing (dilate -> erode)
            maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, kernelCle)
            maska2 = cv2.morphologyEx(maska2, cv2.MORPH_CLOSE, kernelCle)

            # crtanje platforme PLATFORM boundaries
            line1 = np.array([[180, 110], [470, 90], [525, 450], [170, 460], [180, 110]], np.int32).reshape((-1, 1, 2))
            frame = cv2.polylines(frame, [line1], False, (255, 0, 0), thickness=2)

        except:
            # ako nemamo vise frejmova da prikazemo...
            print('EOF')
            print
            'DETEKTOVANO:', personID
            break

            ###############
            #   KONTURE   #
            ###############

        _, contours01, hierarchy = cv2.findContours(maska2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours01:
            area = cv2.contourArea(cnt)
            if (area > areaTHR) & (area < areaMAX):
                #################
                #   PRACENJE    #
                #################
                M = cv2.moments(cnt)
                centerX = int(M['m10'] / M['m00'])
                centerY = int(M['m01'] / M['m00'])
                x, y, w, h = cv2.boundingRect(cnt)

                new = True
                if centerY in range(upLimit, downLimit):
                    for i in persons:
                        if abs(centerX - i.getX()) <= w and abs(centerY - i.getY()) <= h:
                            # objekat je blizu onom sto je prethodno detektovan
                            new = False
                            i.updateCoords(centerX, centerY)  # updejtuje koordinate
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > downLimit:
                                i.setDone()
                            elif i.getDir() == 'up' and i.getY() < upLimit:
                                i.setDone()
                        if i.timedOut():
                            # uklanja osobu sa liste
                            index = persons.index(i)
                            persons.pop(index)
                            del i  # oslobadjamo memoriju

                    if new == True:
                        p = Person.MyPerson(personID, centerX, centerY, maxPersonAge)
                        persons.append(p)
                        personID += 1
                #################
                #   CRTANJE     #
                #################
                cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)

        # END for cnt in contours01

        #############################
        # CRTANJE PUTANJA KRETANJA  #
        #############################
        for i in persons:
            if len(i.getTracks()) >= 2:
                pts1 = np.array(i.getTracks(), np.int32)
                pts1 = pts1.reshape((-1,1,2))
                frame = cv2.polylines(frame,[pts1],False,i.getRGB())
            if i.getId() == 9:
                print
                str(i.getX()), ',', str(i.getY())

        ##############
        #   SLIKE    #
        ##############
        stringDetected = 'DETEKTOVANO: ' + str(personID - 1)

        frame = cv2.polylines(frame, [pts_L1], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L2], False, (255, 255, 255), thickness=1)
        cv2.putText(frame, stringDetected, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, stringDetected, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('Frame', frame)

        # Abort and exit with 'Q' or ESC
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # END while(cap.isOpened())

    f.write(str(videoNaziv) + "," + str(personID - 1) + " \n")


    cap.release()  # release video file
    cv2.destroyAllWindows()  # close all openCV windows