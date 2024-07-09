import cv2
import numpy as np
import time

from rtmpose_utils import preprocess, postprocess, visualize
from rtm_model import PoseModel
from utils import env_info, get_cam
# from mouse import infer
import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.43.229', 8081))

def get_dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def get_angle(pt1, pt2, pt3):
    '''
    计算三点之间的夹角
    '''
    a = get_dist(pt2, pt3)
    b = get_dist(pt1, pt3)
    c = get_dist(pt1, pt2)
    angle = np.arccos((a*a + c*c - b*b) / (2 * a * c))
    return angle


# 用户提前录入的 握鼠标手势的21个关键点坐标
base_21pts = [[305.9375, 105.625 ],
  [380.9375, 155.625 ],
  [437.1875, 216.5625],
  [446.5625, 271.25  ],
  [413.75,   333.75  ],
  [355.9375, 163.4375],
  [349.6875, 222.8125],
  [346.5625, 279.0625],
  [346.5625, 350.9375],
  [290.3125, 161.875 ],
  [268.4375, 238.4375],
  [257.5,    315.    ],
  [257.5,    386.875 ],
  [243.4375, 180.625 ],
  [210.625,  260.3125],
  [210.625,  338.4375],
  [213.75,   407.1875],
  [210.625,  200.9375],
  [190.3125, 258.75  ],
  [185.625,  308.75  ],
  [185.625,  347.8125]]
base_dists = []
# 计算点与点之间的距离，并归一化
for i in range(21):
    for j in range(i+1, 21):
        base_dists.append(get_dist(base_21pts[i], base_21pts[j]))
base_dists = np.array(base_dists)
base_dists = base_dists / np.max(base_dists)

base_angles = []
# 计算关节间的夹角
# 1，2，3；2，3，4；5，6，7；6，7，8；9，10，11；10，11，12；13，14，15；14，15，16；17，18，19；18，19，20
for i in range(0, 18, 3):
    base_angles.append(get_angle(base_21pts[i], base_21pts[i+1], base_21pts[i+2]))
base_angles = np.array(base_angles)


alpha = 0.5
beta = 0.5
loss_thr = 10
def loss(base_21pts, cur_21pts):
    '''
    定义两个21个关键点坐标之间的差距
    '''
    # assert len(base_21pts) == 21
    # assert len(cur_21pts) == 21
    loss = 0

    cur_dists = []
    for i in range(21):
        for j in range(i+1, 21):
            cur_dists.append(get_dist(cur_21pts[i], cur_21pts[j]))
    cur_dists = np.array(cur_dists)
    cur_dists = cur_dists / np.max(cur_dists)

    cur_angles = []
    for i in range(0, 18, 3):
        cur_angles.append(get_angle(cur_21pts[i], cur_21pts[i+1], cur_21pts[i+2]))
    cur_angles = np.array(cur_angles)

    loss = alpha * np.sum(np.abs(base_dists - cur_dists)) + beta * np.sum(np.abs(base_angles - cur_angles))

    return loss

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

env_info()
pose_model = PoseModel()
cam, camid = get_cam()

# 判定用户正在我鼠标时，累计计时(秒)
# 每分钟发送一次计时，之后清空
hold_time = 0
min_start_time = time.time()
last_tick = min_start_time
while True:
    ret, frame=cam.read()
    if not ret:
        continue
    if frame is None:
        continue
    if camid==1:
        frame=cv2.flip(frame,1)

    # print('start pose estimate')
    keypoints, scores = infer(frame)
    # print(keypoints)
    cur_21pts = keypoints[0]

    loss_val = loss(base_21pts, cur_21pts)
    print('loss: ', loss_val)

    tick = time.time()
    if loss_val < loss_thr:
        hold_time += tick - last_tick
        visualize(frame, keypoints, scores, thr=0.3)

    last_tick = tick
    print('time: ', hold_time)


    cv2.putText(frame, 'loss: {:.4f}'.format(loss_val), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(frame, 'holding time: {:.4f}'.format(hold_time), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    msg = hold_time
    client.send(str(msg).encode('utf8'))

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
