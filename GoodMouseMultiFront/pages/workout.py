import time
import streamlit as st
import requests
import pymysql

#  DatabaseConnection
HOST = 'localhost'
PORT = 3306
USER = 'root'
PASSWORD = '123456'
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

st.title("Workout")

st.write("This is the workout page.")

st.subheader("动作一：蛇形伸展")
with st.expander("展开"):
    st.write("请抬起手臂，手臂前伸，尽量保持手腕伸直。")
    st.write("手心朝下，左右摆动手掌。")

    st.video("https://www.bilibili.com/video/BV1Wy4y137Lb?t=154.4")

    progress_bar = st.progress(0)

    if st.button("开始计数", 0):      
        cnt = 0
        state = 0
        while cnt < 20:
            # response = requests.get('http://localhost:5000/get_action')
            # action_data = response.json()
            # action = action_data.get('action')
            # 数据读取
            cursor.execute('SELECT * FROM action_count ORDER BY id DESC LIMIT 1')
            db_conn.commit()
            result:tuple = cursor.fetchone()
            cnt = result[3]
            print(cnt)
            # time.sleep(0.1)
            
            # 计算进度
            i = cnt / 20 * 100
            progress_bar.progress(int(i))
            # time.sleep(1)
            # st.rerun()
            # st.write(f"{cnt} / 20")

        st.write("完成！")

        if st.button("再来一次"):
            progress_bar.progress(0)
            cnt = 0
            state = 0

st.subheader("动作二：握拳张开")
with st.expander("展开"):
    st.write("请缓缓握拳，然后慢慢伸展手指。")

    st.video("https://www.bilibili.com/video/BV1Wy4y137Lb?t=154.4")

    progress_bar = st.progress(0)

    if st.button("开始计数", 1):
        cnt = 0
        state = 0
        while cnt < 20:
            # response = requests.get('http://localhost:5000/get_action')
            # action_data = response.json()
            # action = action_data.get('action')
            # 数据读取
            cursor.execute('SELECT * FROM action_count ORDER BY id DESC LIMIT 1')
            db_conn.commit()
            result:tuple = cursor.fetchone()
            cnt = result[3]
            # 计算进度
            i = cnt / 20 * 100
            progress_bar.progress(int(i))
            st.write(f"{cnt} / 20")
            
            state = 0

        st.write("完成！")

        if st.button("再来一次"):
            progress_bar.progress(0)
            cnt = 0
            state = 0


st.subheader("动作三：手掌后拉")
with st.expander("展开"):
    st.write("小臂向上，用另一只手将手掌拉向自己，保持10秒。")

    st.video("https://www.bilibili.com/video/BV1Wy4y137Lb?t=154.4")

    progress_bar = st.progress(0)

    if st.button("开始计时"):
        for i in range(10):
            progress_bar.progress((i + 1) * 10)
            time.sleep(1)

        st.write("完成！")

        if st.button("再来一次"):
            progress_bar.progress(0)
            cnt = 0
            state = 0
    