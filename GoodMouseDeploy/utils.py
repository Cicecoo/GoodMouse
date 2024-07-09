import time
import subprocess
import cv2 
import aidlite


def env_info():
    # Aidlite SDK版本
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
    
def get_cam():
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
    
    return cap, camid
