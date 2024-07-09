import cv2
import time
import subprocess


class MyCamera():
    def __init__(self):
        self.opened = False
        self.cam = None
        self.camid = None

    def get_camid():
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
        
    def set_camid(self, camid):
        self.camid = camid

    def open(self):
        while not self.opened:
            if self.camid is not None:
                self.cam=cv2.VideoCapture(self.camid, device='mipi')
            else:
                self.camid = self.get_camid()
                print(self.camid)
                if self.camid is None:
                    print("No cam")
                self.cam = cv2.VideoCapture(self.camid)
                self.cam.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))

                self.cam.set(cv2.CAP_PROP_EXPOSURE, 1000)

            if self.cam.isOpened():
                self.opened = True
            else:
                self.cam.release()
                time.sleep(0.5)