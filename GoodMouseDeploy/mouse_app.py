import cv2
import numpy as np
import time

from rtmpose_utils import preprocess, postprocess, visualize
from rtm_model import PoseModel
from utils import env_info, get_cam
from mouse_utils import findDistance, fingersUp, fingersLeft, fingersRight
import socket

# import autopy
##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
#########################

class MyScreen():
    def __init__(self):
        self.size_tuple = (2560, 1440) # (1920, 1080)
    
    def size(self):
        return self.size_tuple

class MyMouse():
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(('192.168.43.229', 8080))

    def move(self, x, y):
        msg = 'mv|' + str(int(x)) + '|' + str(int(y)) + '|'
        self.client.send(msg.encode('utf-8'))
        print('moving mouse to [', str(int(x)), ',', str(int(y)), ']')

    # def click(self):
    #     msg = 'cl|'
    #     self.client.send(msg.encode('utf-8'))
    #     print('click')

    #  鼠标按键操作
    def left_down(self):
        msg = 'ld|'
        self.client.send(msg.encode('utf-8'))
        print('left down')

    def left_up(self):
        msg = 'lu|'
        self.client.send(msg.encode('utf-8'))
        print('left up')

    def right_down(self):
        msg = 'rd|'
        self.client.send(msg.encode('utf-8'))
        print('right down')

    def right_up(self):
        msg = 'ru|'
        self.client.send(msg.encode('utf-8'))
        print('right up')

    def scroll_controll(self, direct):
        msg = 'sc|' + direct + '|'
        self.client.send(msg.encode('utf-8'))
        print('scroll length = '+ direct)

class AutopySim():
    def __init__(self):
        self.screen = MyScreen()
        self.mouse = MyMouse()

myautopy = AutopySim()

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# cap = cv2.VideoCapture(1)
# cap.set(3, wCam)
# cap.set(4, hCam)
# detector = htm.handDetector(maxHands=1)




env_info()

pose_model = PoseModel()

def infer(image):
    resized_img, center, scale = preprocess(image, (256, 256))

    pose_model.interpreter.set_input_tensor(in_tensor_idx=0 , input_data=resized_img.data)
    pose_model.interpreter.invoke()
    simcc_x = pose_model.interpreter.get_output_tensor(0)
    simcc_y = pose_model.interpreter.get_output_tensor(1)
    # print('simcc_x:', simcc_x.shape)
    # (10752,) to (1,21,512)
    simcc_x = simcc_x.reshape(1, 21, 512)
    simcc_y = simcc_y.reshape(1, 21, 512)
    outputs = (simcc_x, simcc_y)
    # first 21 x
    # print('simcc_x:', simcc_x.shape)

    keypoints, scores = postprocess(outputs, pose_model.model_input_size, center, scale)
    return keypoints, scores 

cam, camid = get_cam()

# while True:
#     ret, frame=cam.read()
#     if not ret:
#         continue
#     if frame is None:
#         continue
#     if camid==1:
#         frame=cv2.flip(frame,1)

#     # print('start pose estimate')
#     keypoints, scores = infer(frame)

#     visualize(frame, keypoints, scores, thr=0.3)
    
#     cv2.imshow("frame", frame)
#     cv2.waitKey(1)


wScr, hScr = myautopy.screen.size()
print(wScr, hScr)

while True:
    # 1. Find hand Landmarks
    # success, img = cap.read()
    # img = detector.findHands(img)
    # lmList, bbox = detector.findPosition(img)
    ret, frame=cam.read()
    if not ret:
        continue
    if frame is None:
        continue
    if camid==1:
        frame=cv2.flip(frame,1)

    keypoints, scores = infer(frame)

    keypoints = keypoints[0]
    print(len(keypoints))
    if len(keypoints) < 21:
        continue

    lmList = [] # landmark list
    i = 0
    for point in keypoints:
        lmList.append([i, int(point[0]), int(point[1])])
        # print([i, point[0], point[1]])
        i += 1

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = fingersUp(lmList)

        left = fingersLeft(lmList)
        right = fingersRight(lmList)

        # print(fingers)
        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR),
        (255, 0, 255), 2)
    # 4. Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

    # 7. Move Mouse
    myautopy.mouse.move(wScr - clocX, clocY)
    cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    plocX, plocY = clocX, clocY

    # 8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        # 9. Find distance between fingers
        length, frame, lineInfo = findDistance(lmList, 8, 12, frame)
        print(length)
        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(frame, (lineInfo[4], lineInfo[5]),
            15, (0, 255, 0), cv2.FILLED)
            myautopy.mouse.left_down()
        elif length > 80:
            cv2.circle(frame, (lineInfo[4], lineInfo[5]),
            15, (0, 0, 255), cv2.FILLED)
            myautopy.mouse.left_up()
    
    # # 向下滚动
    # if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
    #     myautopy.mouse.scroll_controll('down')

    # if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
    #     myautopy.mouse.scroll_controll('up')
    # if right[1] == 1 and right[2] == 1:
    #     # cv2.circle(frame, (lineInfo[4], lineInfo[5]),
    #     # 15, (0, 125, 125), cv2.FILLED)
    #     myautopy.mouse.scroll_controll('down')

    # if left[1] == 1 and left[2] == 1:
    #     # cv2.circle(frame, (lineInfo[4], lineInfo[5]),
    #     # 15, (0, 125, 125), cv2.FILLED)
    #     myautopy.mouse.scroll_controll('up')

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # 12. Display
    cv2.imshow("mouse", frame)
    cv2.waitKey(1)
