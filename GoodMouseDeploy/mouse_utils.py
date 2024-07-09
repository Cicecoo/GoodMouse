import math
import cv2


tipIds = [4, 8, 12, 16, 20]

def fingersUp(landmarks):
    fingers = []
    # Thumb
    if landmarks[tipIds[0]][1] > landmarks[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for id in range(1, 5):
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        # totalFingers = fingers.count(1)

    return fingers

def fingersLeft(landmarks):
        fingers = []
        # Thumb
        if landmarks[tipIds[0]][0] > landmarks[tipIds[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if landmarks[tipIds[id]][0] < landmarks[tipIds[id] - 2][0]:
                fingers.append(1)
            else:
                fingers.append(0)
            # totalFingers = fingers.count(1)

        return fingers
    
def fingersRight(landmarks):
        fingers = []
        # Thumb
        if landmarks[tipIds[0]][0] < landmarks[tipIds[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if landmarks[tipIds[id]][0] > landmarks[tipIds[id] - 2][0]:
                fingers.append(1)
            else:
                fingers.append(0)
            # totalFingers = fingers.count(1)

        return fingers

def findDistance(landmarks, p1, p2, img, draw=True,r=15, t=3):
    x1, y1 = landmarks[p1][1:]
    x2, y2 = landmarks[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]