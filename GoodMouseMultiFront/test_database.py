import pymysql
import time
import random

#  DatabaseConnection
HOST = 'localhost'
PORT = 3306
USER = 'root'
PASSWORD = 'root'
CHARSET = 'utf8mb4'
# DATABASE = "test"
DATABASE = "goodmouse_data"

#  Initialize MySQL connection
db_conn = pymysql.connect(
    host=HOST,		# 主机名（或IP地址）
    port=PORT,				# 端口号，默认为3306
    user=USER,			# 用户名
    password=PASSWORD,	    # 密码
    charset=CHARSET  		# 设置字符编码
)
cursor = db_conn.cursor()
db_conn.select_db(DATABASE)
count = 0
date = 1
while(date<10):
    # time.sleep(1)
    # if count == 10:
    #     count = 0
    # else:
    #     count += 1
    # update_count = str(count)
    # sql = 'UPDATE action_count SET count = '+ update_count + ' WHERE id = 4'
    str1 = "2024-07-0"
    str2 = " 00:00:00"
    datetime = str1 + str(date) + str2
    print(datetime)
    date += 1
    seconds = random.uniform(0, 10000)
    sql = 'INSERT INTO mouse_usage VALUES '+ '(\'' + datetime + '\',' + str(seconds) +')'
    print(sql)
    cursor.execute(sql)
    db_conn.commit()