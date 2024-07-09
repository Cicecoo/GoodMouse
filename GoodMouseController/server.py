import socket
import threading
from mouse import GM
IP_PORT = ('192.168.43.229',8080)
BUFFER_SIZE = 1024
CLIENT_NUM = 1


    # def move(self, x, y):
    #     msg = 'mv|' + str(x) + '|' + str(y) + '|'
    #     client.send(msg.encode('utf-8'))
    #     print('moving mouse to [', x, ',', y, ']')

    # def click(self):
    #     msg = 'dc|'
    #     client.send(msg.encode('utf-8'))
    #     print('click')



class GoodMouseServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(IP_PORT)
        self.sock.listen(CLIENT_NUM)
        self.Is_disconnected = True
        self.supported_code = ['mv', 'ld', 'lu', 'rd', 'ru', 'sc']

    def run(self):  # 网页端主线程
        while self.Is_disconnected:
            con, addr = self.sock.accept()
            self.Is_disconnected = False
            while not self.Is_disconnected:
                try:
                    msg = con.recv(BUFFER_SIZE)
                    # 这里添加解码和控制
                    # print(msg)
                    code_list = msg.decode('utf-8').split('|')
                    if code_list[0] in self.supported_code:
                        GM.set_control_state('enabled')
                        # print(GM.control_state)

                    if code_list[0] == 'mv':  # 移动鼠标
                        GM.set_position(int(code_list[1]), int(code_list[2]))
                        print('moving mouse to [', code_list[1], ',', code_list[2], ']')
                        # GM.button_control('left', 'up')
                        # GM.button_control('right', 'up')

                    elif code_list[0] == 'ld':  # 左键按下
                        # print('left down')
                        GM.button_control('left', 'down')
                    elif code_list[0] == 'lu':  # 左键抬起
                        # print('left up')
                        GM.button_control('left', 'up')
                    elif code_list[0] == 'rd':  # 右键按下
                        # print('right down')
                        GM.button_control('right', 'down')
                    elif code_list[0] == 'ru':  # 右键抬起
                        # print('right up')
                        GM.button_control('right', 'up')

                    # elif code_list[0] == 'sc':  # 滚轮
                    #     print('scroll')
                    #     if code_list[1] == 'up':
                    #         GM.scroll_len = -120
                    #     elif code_list[1] == 'down':
                    #         GM.scroll_len = 120
                    #     else:
                    #         GM.scroll_len = 0
                    #     print('scroll: ', GM.scroll_len)


                    # 这里添加解码和控制
                except Exception as e:
                    print(e)
                    self.Is_disconnected = True
                    break


GMServer = GoodMouseServer()
GMServer.start()

