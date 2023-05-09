import cv2
import time
import math
import itertools
import mediapipe as mp

# Create a class for button
class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                        (225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                        (50, 50, 50), 3)
        cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 65), cv2.FONT_HERSHEY_PLAIN,
                    2, (50, 50, 50), 2)

    def checkClick(self, img, x, y):
        if self.pos[0] < x < self.pos[0] + self.width and \
                self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img, (self.pos[0] + 3, self.pos[1] + 3),
                            (self.pos[0] + self.width - 3, self.pos[1] + self.height - 3),
                            (255, 255, 255), cv2.FILLED)
            cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 80), cv2.FONT_HERSHEY_PLAIN,
                        5, (0, 0, 0), 5)
            return True
        else:
            return False


# Buttons
buttonListValues = [['7', '8', '9', '*'],
                    ['4', '5', '6', '-'],
                    ['1', '2', '3', '+'],
                    ['0', '/', '.', '=']]
buttonList = []
for x, y in itertools.product(range(4), range(4)):
    xpos = x * 100 + 600
    ypos = y * 100 + 150

    buttonList.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))


# Define detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Active camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variables
myEquation = ''

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        success, image = cap.read()

        # Horizontal flipping
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB for accurate results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mark the image as not writeable to pass by reference for accurate results
        image.flags.writeable = False

        # Process the image and detect the pose
        results = hands.process(image)

        # remark the image as writeable
        image.flags.writeable = True

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw All
        cv2.rectangle(image, (600, 50), (600 + 400, 50 + 100),
                        (225, 225, 225), cv2.FILLED)

        cv2.rectangle(image, (600, 50), (600 + 400, 50 + 100),
                        (50, 50, 50), 3)
        for button in buttonList:
            button.draw(image)

        # Write the Final answer
        cv2.putText(image, myEquation, (610, 130), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 0, 0), 3)


        # Draw landmarks
        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get landmarks list
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        if lmList:
            # print(lmList[4], lmList[8])

            x1, y1 = lmList[8][1], lmList[8][2]
            x2, y2 = lmList[12][1], lmList[12][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)

            x, y = lmList[8][1], lmList[8][2]

            # If clicked check which button and perform action
            if length < 50 :
                for i, button in enumerate(buttonList):
                    if button.checkClick(image, x, y):
                        myValue = buttonListValues[int(i % 4)][int(i / 4)]  # get correct number
                        if myValue == '=':
                            myEquation = str(eval(myEquation))
                        else:
                            myEquation += myValue
                            time.sleep(0.2)

        # Get current time
        current_time = str(time.ctime())
        cv2.putText(image, current_time, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        # Show results
        cv2.imshow('Live', image)

        # To exit from live, press Esc key
        if cv2.waitKey(1) & 0xFF == 27: # 27 is the Esc Key
            break

# release camera and close windows
cap.release()
cv2.destroyAllWindows()