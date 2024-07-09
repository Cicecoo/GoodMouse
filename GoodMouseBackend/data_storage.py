# 向服务器存储数据

import socket
import threading
import pymysql
import time

#  TCPServer
IP_PORT = ('192.168.43.229',8081)
BUFFER_SIZE = 1024
CLIENT_NUM = 1
#  DatabaseConnection
HOST = 'localhost'
PORT = 3306
USER = 'root'
PASSWORD = 'root'
CHARSET = 'utf8mb4'
DATABASE = "goodmouse_data"

class DataBaseServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(IP_PORT)
        self.sock.listen(CLIENT_NUM)
        self.Is_disconnected = True
        self.record_hour = time.localtime().tm_hour
        self.using_seconds = 0
        # 建立数据库连接
        self.db_conn = pymysql.connect(
            host=HOST,		# 主机名（或IP地址）
            port=PORT,				# 端口号，默认为3306
            user=USER,			# 用户名
            password=PASSWORD,	    # 密码
            charset=CHARSET  		# 设置字符编码
        )
        self.cursor = self.db_conn.cursor()
        self.db_conn.select_db(DATABASE)
        print("Database connected")

    def run(self):  # 主线程
        while self.Is_disconnected:
            con, addr = self.sock.accept()
            self.Is_disconnected = False
            while not self.Is_disconnected:
                try:
                    msg = con.recv(BUFFER_SIZE) # 直接发秒数
                    msg = msg.decode('utf-8')
                    msg = float(msg)

                    print(msg)

                    self.using_seconds += msg
                    # code_list = msg.decode('utf-8').split('|')
                    # if code_list[0] == 'ct':
                    #  TODO 通信格式未定
                    local_time_now = time.localtime()
                    # if self.record_hour != local_time_now :
                    if True:
                        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_now)
                        formatted_time = '\'' + formatted_time + '\''
                        self.insert_data("mouse_usage", formatted_time+', '+str(self.using_seconds))
                        self.record_hour = local_time_now
                        self.using_seconds = 0
                except Exception as e:
                    print(e)
    # (1064, "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '17:20:47,60.0)' at line 1")


    def insert_data(self, table_args, data):  #  table_args格式为"表名称 (参数1,参数2,...)" 数据为元组
        # data = ",".join(data)
        sql = "INSERT INTO " + table_args + " VALUES (" + data + ")"

        print(sql)

        self.cursor.execute(sql)
        self.db_conn.commit()
    
db_server = DataBaseServer()
db_server.start()