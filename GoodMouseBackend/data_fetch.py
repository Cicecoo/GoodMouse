# 从数据库取数据，由flask转发给前端

from flask import Flask, jsonify
import pymysql

app = Flask(__name__)

#  DatabaseConnection
HOST = 'localhost'
PORT = 3306
USER = 'root'
PASSWORD = 'root'
CHARSET = 'utf8mb4'
DATABASE = "goodmouse_data"
# DATABASE = "test"

db_conn = pymysql.connect(
    host=HOST,		        # 主机名（或IP地址）
    port=PORT,				# 端口号，默认为3306
    user=USER,			    # 用户名
    password=PASSWORD,	    # 密码
    charset=CHARSET  		# 设置字符编码
)
cursor = db_conn.cursor()
db_conn.select_db(DATABASE)

# API路由 - 获取今日使用鼠标时间
@app.route('/api/today_mouse_usage', methods=['GET'])
def get_today_mouse_usage():
    # 编写查询今日使用鼠标时间的SQL语句
    query = "SELECT SUM(using_seconds) FROM mouse_usage WHERE DATE(time_stamp) = CURDATE()"
    cursor.execute(query)
    result = cursor.fetchone()[0]
    return jsonify({'today_mouse_usage': result})

# API路由 - 获取每日使用鼠标时间
@app.route('/api/daily_mouse_usage', methods=['GET'])
def get_daily_mouse_usage():
    # 编写查询每日使用鼠标时间的SQL语句
    query = "SELECT DATE(time_stamp) AS day, SUM(using_seconds) AS total_mouse_time FROM mouse_usage GROUP BY DATE(time_stamp)"
    cursor.execute(query)
    results = cursor.fetchall()
    daily_usage = [{'day': row[0], 'total_mouse_time': row[1]} for row in results]
    return jsonify({'daily_mouse_usage': daily_usage})

if __name__ == '__main__':
    app.run(debug=True)
