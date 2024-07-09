import pyautogui as mouse
import time
import threading

class GoodMouse(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.mouse = mouse
        (self.Max_X, self.Max_Y) = self.mouse.size()
        print("Screen Size is", self.Max_X, "x", self.Max_Y)
        self.mouse.PAUSE = 0.01  # 设置每次调用后的暂停时间
        self.mouse.FAILSAFE = False  # 鼠标移动到左上角触发错误中断
        (self.x_position, self.y_position) = self.mouse.position()
        self.scroll_len = 0  # 滚轮滚动距离
        self.button = 'none'  # 按键 有 'left' 'right' 'middle'
        self.bt_state = 'up'  # 按键状态 'up' 'down' 'double_click'
        self.bt_last_state = self.bt_state
        self.isWorking = True  # 工作状态

        self.control_state = 'none'  # 控制状态 'none' 'mouse' 'keyboard'

    def run(self):  # 控制鼠标主线程
        while self.isWorking:
            #  实时刷新坐标
            # print(self.control_state)
            if self.control_state == 'enabled':
                print(self.x_position, self.y_position)
                self.mouse.moveTo(self.x_position, self.y_position)
                self.set_control_state('none')

            #  点击判定
            if self.button == 'left' or self.button == 'right' or self.button == 'middle':
                if self.bt_last_state != 'up' and self.bt_state == 'up':
                    print(self.button, self.bt_state)
                    self.mouse.mouseUp(self.x_position, self.y_position, button=self.button)
                    # self.button = 'none'  # 维持按键状态
                    self.bt_last_state = self.bt_state
                elif self.bt_last_state == 'up' and self.bt_state == 'down':
                    print(self.button, self.bt_state)
                    self.mouse.mouseDown(self.x_position, self.y_position, button=self.button)
                    # self.button = 'none'  # 维持按键状态
                    self.bt_last_state = self.bt_state
                    
                elif self.bt_state == 'double_click':
                    print(self.button, self.bt_state)
                    self.mouse.doubleClick(self.x_position, self.y_position, button=self.button)
                    # self.button = 'none'  # 维持按键状态
                    self.bt_last_state = self.bt_state
                    
            else:
                pass

            #  实时刷新滚轮数据
            self.mouse.scroll(self.scroll_len)
            if self.scroll_len != 0:  # 清空滚轮数据
                self.scroll_len = 0

    # 以下为提供给其他模块的接口
    def set_position(self, x_pos, y_pos):  # 移动光标接口
        self.x_position = x_pos
        self.y_position = y_pos

    def button_control(self, button, state):  # 按键接口
        self.button = button
        self.bt_state = state

    def set_control_state(self, state):  # 按键接口
        self.control_state = state

    def test_func(self):
        (x, y) = (0, 0)
        self.button = 'left'
        self.bt_state = 'down'
        while x < (self.Max_X-10):
            print("test running")
            x = x+1
            y = self.Max_Y/self.Max_X*x
            time.sleep(0.01)
            print("Set Position Is", x, y)
            self.set_position(x, y)
        print("test over")


GM = GoodMouse()
GM.start()
