import cv2
import numpy as np

from rtmpose_utils import preprocess, postprocess, visualize
from rtm_model import PoseModel
from utils import env_info, get_cam


env_info()

# TODO 需要设置模型编号？
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
    # keypoints: 形状为[K, 2]
    return keypoints, scores 

cam, camid = get_cam()

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
    print(keypoints)

    visualize(frame, keypoints, scores, thr=0.3)
    
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
