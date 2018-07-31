# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:51:31 2018

@author: tagombos
"""

import numpy as np
# import imutils
import cv2
from collections import deque
import csv

import re
import copy

"""Globale variabler"""

# Data som kan bli brukt/brukes, for tracking og data til klassifikator
storeHeight = []
storeWidth = []
storeCoordinates = []
storeHu = []
checkPattern = deque(maxlen=4)

storeData = [0, 0]
data = []

treerkast = 0
firerkast = 0
enerkast = 0
Score = 0
Class = 0

checkYdir = (deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),)

# Her går bunn til topp fra venstre til høyre.

store_throwWidth = []
checkXdir = (deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),
             deque(maxlen=2),)
VideoFile = 'test/3test.mov'
# VideoFile = 'Data/video/show2.mov'

text_file = np.loadtxt('Trening1750samples.txt', dtype=np.float32)

k = np.arange(2)
train_labels = np.repeat(k, 880)[:, np.newaxis].astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(text_file, cv2.ml.ROW_SAMPLE, train_labels)


# Stream er mainfunksjonen
def stream():
    throwCounter = 0
    numOfObjects = 0
    # Velger hvilket kamera eller film
    cap = cv2.VideoCapture(VideoFile)

    # Dette er HSV terskelene for å filtere ut bakgrunn.
    # Vi har brukt programmet rangeDetector.py til å manuelt finne gode verdier

    #    Red
    #    colorLower = (0, 138, 159)
    #    colorUpper = (213, 255, 255)
    # red
    colorLower = (166, 180, 86)
    colorUpper = (213, 255, 255)

    #    Testsett colour
    #    colorLower = (93, 132, 88)
    #    colorUpper = (255, 255, 255)

    #    #Green
    #    colorLower = (27, 110, 68)
    #    colorUpper = (95, 255, 255)
    #
    # Green - Live kontor
    #    colorLower = (134, 176, 16)
    #    colorUpper = (199, 255, 255)

    # =============================================================================
    #     Initialisering av globale variabler
    # =============================================================================
    # ptsTot er ett sett med deque tuples som vil lagre hvert koordinat til objektene våre.
    # Max 7 objekter av ganger foreløpig
    mlen = 45
    ptsTot = (deque(maxlen=mlen),
              deque(maxlen=mlen),
              deque(maxlen=mlen),
              deque(maxlen=mlen),
              deque(maxlen=mlen),
              deque(maxlen=mlen),
              deque(maxlen=mlen))

    # Teller oppover for hver frame
    counter = 0

    # Initialiserer dequene våre på forhånd
    for ii in range(0, len(ptsTot)):
        for inpt in range(0, 30):
            ptsTot[ii].appendleft((0, 0))

    # =============================================================================
    #             While-løkka starter
    # =============================================================================

    while (1):

        # Initialiserer pts som vil lagre koordinatene til ballene våre i denne framen.
        pts = []

        # Leser en frame fra kameraet/videoen
        _, frame = cap.read()
        #

        # Her blures først bildet før det blir konvert til till HSV verdier
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Lager en maske for bildet innenfor HSV grensene vi satte i starten av programmet.
        # Deretter eroder vi bildet for å fjære små korn som er igjen.
        # Etterpå dilates det slik at konturene av objektet blir litt kraftigere
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, 2)
        mask = cv2.dilate(mask, None, 5)

        # Finner konturene i masken vår
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Går kun videre hvis det faktisk finnes noen konturer
        if len(cnts) > 0:

            # Finner den største konturen og definerer ett threshold som er 40% av den største konturen
            # Dette brukes til æ filtrere bort små blobs og kune fokusere på de store konturene
            c = max(cnts, key=cv2.contourArea)
            # tåler 40% minimum størrelse på konturen
            objectSizeThreshold = np.size(c) * 0.4

            # Initialiserer våre kommende center for ballene
            center = None

            # Beregner og skriver inn antall objekter på skjermen.
            numOfObjects = len([1 for ii in cnts if np.size(ii) > objectSizeThreshold])
            screenText = str(numOfObjects) + ' objects on screen'

            cv2.putText(frame, screenText, (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 0, cv2.LINE_AA)
            # Går gjennom hver kontur og finner senterpunktet vedhjelp av cv2.moments
            for element in cnts:
                if np.size(element) > objectSizeThreshold:
                    # Tegner sirkel rundt detektert objekt
                    ((x, y), radius) = cv2.minEnclosingCircle(element)
                    M = cv2.moments(element)
                    # Beregner center koordinat
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # Hvis radius er stor er stor nok tegner vi sirkel rundt objektene våre.
                    # En liten i midten, og en større rundt objektet

                    if radius > 5:
                        cv2.circle(frame, center, 5, (0, 255, 255), -1)
                        cv2.circle(frame, (int(x), int(y)), int(radius) + 2, (0, 255, 255), 2)

                    # Hvert senter i framen blir lagret i pts
                    pts.append(center)

                    collectData(ptsTot, counter, frame, numOfObjects, radius, checkYdir, throwCounter)
            # Tar for seg hvert kordinat "i" en nye frame.
            # Deretter sammenlikner den koordinatet med hvert av de gamle koordinatene i forrige frame
            # Den vil så finne ut hvilken av de gamle som var nærmest og sørge for at kooridnatene
            # blir lagret i samme deque
            k = 0
            for i in pts:

                # Initaliserer distansen mellom hvert senter
                dist = []
                # Looper gjennom hver av koordinatene i den gamle framen
                for j in range(0, len(ptsTot)):
                    # Finner distansen mellom det nye koordinatet vårt og til de 3 gamle.
                    # Så finner vi indeksen hvor dette koordinatet ligger
                    dist.append(np.linalg.norm
                                (i[0] - ptsTot[j][0][0]) + np.linalg.norm(i[1] - ptsTot[j][0][1]))
                minDistIdx = np.argmin(dist)

                # If settning som skal filtrere bort avstander som er for store.
                # dvs. feil klassifisering
                if dist[minDistIdx] < 150:
                    # Tar det gjeldende koordinatet vi tester for å plasserer det i riktig deque
                    ptsTot[minDistIdx].appendleft(i)

                    # Hvis sdet ikke har er noe objekt så ikke tegn noe
                    for j in range(1, len(ptsTot[minDistIdx])):
                        if ptsTot[minDistIdx][j - 1] == None or ptsTot[minDistIdx][j] == None:
                            continue
                        thickness = int(np.sqrt(25 / float(j + 1)) * 2)
                        cv2.line(frame, ptsTot[minDistIdx][j], ptsTot[minDistIdx][j - 1], (0, 0, 255), thickness)

            # Lagrer de første koordinatene våre i ptsTot
            if ptsTot[0][0] == (0, 0):
                for ii in range(0, numOfObjects):
                    ptsTot[ii].appendleft(pts[ii])


        # Viser bildet
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

        # =============================================================================
        #         Press "esc" for å avslutte
        # =============================================================================

        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break
        counter += 1

    # Avslutter bildeprosessen
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)


# =============================================================================
# Funksjonene våre
# =============================================================================


# =============================================================================
# #Denne blir implementert i funksjonen drawTail()
# =============================================================================
def findHuMoments(image, frame):
    global treerkast
    global firerkast
    global Score
    # Segementerer den tegnede kurven til ballen og finner så Hu-momentene
    mask_Hu = cv2.inRange(image, (0, 0, 254), (0, 0, 255))

    cv2.imshow('plot', mask_Hu)

    # Finner momentene til kurven. Så finner den Hu-momentene
    M = cv2.moments(mask_Hu)
    Hu = cv2.HuMoments(M).flatten()
    # Man må visst ta log av Hu-momentene for best resultat
    Hu = -np.sign(Hu) * np.log10(np.abs(Hu))

    # =============================================================================
    #     Bruker knn på kastet og sammenlikner med treningssettet. Deretter klassifiserer den kastet
    # =============================================================================
    cnvToKNNformat = [[element for element in Hu]]
    testSet = np.asarray(cnvToKNNformat, dtype=np.float32)

    ret, Class, neighbours, dist = knn.findNearest(testSet, 15)
    print(Class)
    if Class == [[0.]]:
        treerkast += 1
    if Class == [[1.]]:
        firerkast += 1

    Feilbudsjett = (firerkast / (firerkast + treerkast)) * 100
    Score = 'Score: ' + str(int(Feilbudsjett)) + '%'

    # =============================================================================
    #     #Laster inn treningssett og bruker regex til å hente ut ren data innenfor [---]
    # =============================================================================
    #    thefile = open('C:/Users/tagombos/Desktop/Python Code/kjeller-prosjekt/test.txt', 'w')
    #    Hu = re.sub("[\[\]\\n]", "", str(Hu))
    #
    #    storeHu.append(Hu)
    #    for item in storeHu:
    #        thefile.write("%s\n" % item)
    return Class


# =============================================================================
# Plotter banen til kastet ett er det er nådd topppunktet
# =============================================================================
def drawTail(ptsTot, image, numOfObjects, counter):
    s = copy.deepcopy(image)
    for j in range(1, len(ptsTot)):

        # Hvis sdet ikke har er noe objekt så ikke tegn noe
        if ptsTot[j - 1] == (0, 0) or ptsTot[j] == (0, 0):
            continue
        thickness = 5

        cv2.line(s, ptsTot[j], ptsTot[j - 1], (0, 0, 255), thickness)

    return s


# =============================================================================
# CollectData() bergener topppunkt og bunnpunkt for kastene og hastigheten (endring fra frame til frame)
# Den har en logikk som vet når ett kast er gjort, og beregner så Humomentene fra plottet til kastet.
# =============================================================================
def collectData(ptsTot, counter, frame, numOfObjects, radius, checkYdir, throwCounter):
    global Class
    yret = 0

    for i in range(0, numOfObjects):

        # Finner topp og bunnpunkt for kastene
        maxHeight = min(ptsTot[i], key=lambda item: item[1])
        minHeight = max(ptsTot[i], key=lambda item: item[1])

        # Tegner topppunktene i banen
        if maxHeight and minHeight is not (0, 0):
            cv2.circle(frame, maxHeight, 5, (255, 0, 255), -1)
            cv2.circle(frame, minHeight, 5, (255, 0, 255), -1)

        # sammenlikner x,y koordinat i nåværende frame mot x,y koordinat for 5 og 2 frames siden
        # for å beregne en hastighet dx, dy for hastighet i x og y retning
        if counter > 20 and [-1] is not None:

            dx = (ptsTot[i][5][0] - ptsTot[i][0][0])
            dy = (ptsTot[i][5][1] - ptsTot[i][0][1])

            cv2.putText(frame, "Ball {}- dx: {}, dy: {}".format(i + 1, dx, dy), (10, 10 + yret),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            checkYdir[i].appendleft(dy)
            checkXdir[i].appendleft(dx)

            #            Lagrer topp- og bunnpunkt når ballen skifter retning i x eller y.
            if counter > 50:
                # =============================================================================
                #                 Sjekker x rettning.
                # =============================================================================
                if checkXdir[i][0] < 0 and checkXdir[i][1] > 0:
                    # Finn topppunkt kordinater
                    maxWidth = np.amax(ptsTot[i], 0)
                    minWidth = np.amin(ptsTot[i], 0)

                    # Finn avstanden mellom dem
                    throwWidth = (maxWidth[0] - minWidth[0]) / radius
                    storeWidth.append(throwWidth)

                if checkXdir[i][0] > 0 and checkXdir[i][1] < 0:
                    # Finn topppunkt kordinater
                    maxWidth = np.amax(ptsTot[i], 0)
                    minWidth = np.amin(ptsTot[i], 0)

                    # Finn avstanden mellom dem
                    throwWidth = (maxWidth[0] - minWidth[0]) / radius
                    storeWidth.append(throwWidth)

                # =============================================================================
                #                 Sjekker y retning. Det er her den definerer att et kast er blitt gjort ved å se på
                #                 når ballen skifter y retning.
                # =============================================================================

                # Får lagret begge to på en gang
                if checkYdir[i][0] < 0 and checkYdir[i][1] > 0:
                    s = drawTail(ptsTot[i], frame, numOfObjects, counter)
                    Class = findHuMoments(s, frame)
                    throwHeight = (minHeight[1] - maxHeight[1]) / radius
                    storeHeight.append(throwHeight)

            yret += 15

    # Skriver antall kast, type kast og classifier score på skjermen.
    if len(storeHeight) > 0:
        cv2.putText(frame, "Throws: {}".format(len(storeHeight)), (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                    1)

        if Class == [[0.]]:
            cv2.putText(frame, 'Throw type: 3', (10, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        elif Class == [[1.]]:
            cv2.putText(frame, 'Throw type: 4', (10, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    if Score != 0:
        cv2.putText(frame, Score, (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    (10, 410)


stream()
















