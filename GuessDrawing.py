import cv2
import mediapipe as mp
import math
import pytesseract
import threading
import numpy as np

cap = cv2.VideoCapture(0)
DrawPoints = []
Connected = False
pytesseract.pytesseract.tesseract_cmd = 'D:\\Tesseract\\tesseract.exe'
frame = 0
drawColor = (255, 0, 255)
CharDetector = None


class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=False):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # ,
                    #                                                self.mpHands.HAND_CONNECTIONS
                    self.mpDraw.draw_landmarks(img, handLms)
        return img

    def findPosition(self, img, handNo=0, draw=False):

        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH

            # if draw:
            #     cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
            #                   (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
            #                   (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):

        if self.results.multi_hand_landmarks:
            myHandType = self.handType()
            fingers = []

            if myHandType == "Left":
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):

        if self.results.multi_hand_landmarks:
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
                # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [x1, y1, x2, y2, cx, cy]

    def handType(self):

        if self.results.multi_hand_landmarks:
            if self.lmList[17][1] < self.lmList[5][1]:
                return "Left"
            else:
                return "Right"


class DetectCharacters:

    def __init__(self, image):
        self.r = {'text': [],
                  'boxes': [],
                  'percentage': [],
                  'str': ''}
        self.Height, self.Width = image.shape[:2]
        self.blank_image = np.zeros((self.Height, self.Width, 3), np.uint8)

    def DetectText(self, points, show=True):
        self.blank_image = np.zeros((self.Height, self.Width, 3), np.uint8)
        for Connections in points:
            for count in range(len(Connections)):
                x = Connections[count][0]
                y = Connections[count][1]
                cv2.circle(self.blank_image, (x, y), 5, (255, 255, 255), cv2.FILLED)
                if count != 0:
                    x2 = Connections[count - 1][0]
                    y2 = Connections[count - 1][1]
                    cv2.line(self.blank_image, (x, y), (x2, y2), (255, 255, 255), 5)

        result = {'text': [],
                  'boxes': [],
                  'percentage': [],
                  'str': ''}
        data = pytesseract.image_to_data(cv2.cvtColor(self.blank_image, cv2.COLOR_BGR2RGB), 'eng')
        data = data.splitlines()[1:]
        if data:
            for i in range(len(data)):
                temp = data[i].split('\t')
                if float(temp[10]) > 0:
                    result['str'] += temp[11] + ' '
                    result['text'].append(temp[11])
                    result['boxes'].append(temp[6:10])
                    result['percentage'].append(temp[10])
        if result['boxes']:
            for i in range(len(result['boxes'])):
                b = result['boxes'][i]
                x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cv2.rectangle(self.blank_image, (x, y), (w + x, h + y), (50, 50, 255), 2)
        print(result)

        self.r = result


detector = HandDetector(detectionCon=0.8, maxHands=1, minTrackCon=.1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 2)
    if not CharDetector:
        CharDetector = DetectCharacters(img)

    if frame % 30 == 0:
        t = threading.Thread(target=CharDetector.DetectText, args=[DrawPoints, ])
        t.start()
        frame = 0

    frame += 1
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if lmList:
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)
        cv2.putText(img, f'Fingers Up: {totalFingers}', (bbox[0], bbox[1] - 90),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

        if totalFingers == 0:
            DrawPoints.clear()
            Connected = False

        myHandType = detector.handType()
        cv2.putText(img, f'Hand: {myHandType}', (bbox[0], bbox[1] - 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

        distance, img, info = detector.findDistance(4, 8, img)
        cv2.putText(img, f'Dist: {int(distance)}', (bbox[0], bbox[1] - 60),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

        if distance <= 30:
            points = (info[4], info[5])
            if not Connected:
                DrawPoints.append([points])
                Connected = True
            else:
                DrawPoints[len(DrawPoints) - 1].append(points)
        else:
            Connected = False

    for Connections in DrawPoints:
        for count in range(len(Connections)):
            x = Connections[count][0]
            y = Connections[count][1]
            cv2.circle(img, (x, y), 5, drawColor, cv2.FILLED)
            if count != 0:
                x2 = Connections[count - 1][0]
                y2 = Connections[count - 1][1]
                cv2.line(img, (x, y), (x2, y2), drawColor, 5)

    cv2.putText(img, "Best Guess: " + str(CharDetector.r['str']), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    width = int(img.shape[1] * 3)
    height = int(img.shape[0] * 3)

    img = cv2.resize(img, (width, height))

    cv2.imshow("Pass Through View", img)
    cv2.imshow("What The Computer Can See", CharDetector.blank_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
