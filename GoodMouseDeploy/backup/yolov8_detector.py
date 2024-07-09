import cv2
import time 
import subprocess
import numpy as np
from blazeface import *
import aidlite

from rtmpose_utils import *
from yolov8_utils import yolo_postprocess, yolo_preprocess, preprocess_image_for_tflite
# from tmp import *
from rtm_model import DetModel, PoseModel


# 检查SDK版本
print(f"Aidlite library version : {aidlite.get_library_version()}")
print(f"Aidlite Python library version : {aidlite.get_py_library_version()}")

# 初始化log
# aidlite.set_log_level(aidlite.LogLevel.INFO)
# aidlite.log_to_stderr()
# aidlite.log_to_file("./fast_SNPE_inceptionv3_")

def get_cap_id():
    try:
        # 构造命令，使用awk处理输出
        cmd = "ls -l /sys/class/video4linux | awk -F ' -> ' '/usb/{sub(/.*video/, \"\", $2); print $2}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split()

        # 转换所有捕获的编号为整数，找出最小值
        video_numbers = list(map(int, output))
        if video_numbers:
            return min(video_numbers)
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# TODO 需要设置模型编号？
det_model = DetModel()

aidlux_type="root"
camid=1
opened = False
while not opened:
    if aidlux_type == "basic":
        cap=cv2.VideoCapture(camid, device='mipi')
    else:
        camid = get_cap_id()
        print(camid)
        if camid is None:
            print("No cam")
        cap = cv2.VideoCapture(camid)
        cap.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_EXPOSURE, 1000)
    if cap.isOpened():
        opened = True
    else:
        cap.release()
        time.sleep(0.5)

while True:
    ret, frame=cap.read()
    if not ret:
        continue
    if frame is None:
        continue
    if camid==1:
        frame=cv2.flip(frame,1)

    print('start det')
    # resized_img, img_height, img_width = yolo_preprocess(frame, 256, 256)
    resized_img = preprocess_image_for_tflite(frame)

    # cv2.imshow('resized', resized_img)
    # cv2.waitKey(1)

    det_model.interpreter.set_input_tensor(in_tensor_idx=0 , input_data=resized_img.data)
    det_model.interpreter.invoke()
    # outputs = det_model.interpreter.get_output_tensor(0).reshape(1, 5, 1344)
    outputs = det_model.interpreter.get_output_tensor(0).reshape(73, 5)

    print(outputs[0])

    # yolo_postprocess(frame, outputs, 256, 256, 640, 480)

    # cv2.imshow("frame", frame)
    # cv2.waitKey(1)
