import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import pymysql
import datetime

# 后端API地址
backend_url = 'http://localhost:5000'

# 获取每天和每小时的鼠标使用时间数据
def fetch_data(time_period):
    if time_period == 'daily':
        response = requests.get(f'{backend_url}/api/daily_mouse_usage')
        # cursor.execute('SELECT * FROM mouse_usage')
        # db_conn.commit()
        # result = cursor.fetchall()
    # elif time_period == 'hourly':
    #     response = requests.get(f'{backend_url}/api/hourly_mouse_usage')
    # else:
    #     return None
    # return result
    return response.json()[f'{time_period}_mouse_usage']


# Streamlit 应用
st.title('Mouse Usage Dashboard')

# 今日鼠标使用时间
st.subheader('今日鼠标使用时间')
# 获取今日使用鼠标时间
# now_date = str(datetime.date.today())
# cursor.execute('SELECT SUM(using_seconds) FROM mouse_usage WHERE DATE(time_stamp)='+now_date)
# db_conn.commit()
response_today = requests.get(f'{backend_url}/api/today_mouse_usage')
today_mouse_usage = response_today.json()['today_mouse_usage']

# print(type(today_mouse_usage))
today_mouse_usage = today_mouse_usage/60

# today_mouse_usage = cursor.fetchone()
# st.metric(label='Today\'s Mouse Time', value=today_mouse_usage, delta=0, 
time_unit = "minutes"  # 单位
formatted_time_value = f"{today_mouse_usage:.2f}"
st.metric(label=f"Today\'s Mouse Time ({time_unit})", value=formatted_time_value, delta=0)

# 历史鼠标使用时间
st.subheader('历史鼠标使用时间')
# time_period = st.radio("Select Time Period", ('daily', 'hourly'))

# time_period = 'daily'
# mouse_usage_data = fetch_data(time_period)
# if mouse_usage_data is not None:
#     # 将数据转换为DataFrame
#     df_mouse_usage = pd.DataFrame(mouse_usage_data)
    
#     if time_period == 'daily':
#         df_mouse_usage['day'] = pd.to_datetime(df_mouse_usage['day']).dt.strftime('%Y-%m-%d')
#         x_label = 'day'
#     # elif time_period == 'hourly':
#     #     df_mouse_usage['hour'] = pd.to_datetime(df_mouse_usage['hour'])
#     #     x_label = 'Hour'

#     # 显示柱形图
#     st.subheader(f'{time_period.capitalize()} Mouse Usage')
#     df_mouse_usage.set_index(x_label)
#     date = df_mouse_usage['day'].values.tolist()
#     usingtime = df_mouse_usage['total_mouse_time'].values.tolist()
#     print(usingtime)
#     chartdata = pd.DataFrame(usingtime, date)
#     print(chartdata)
#     st.bar_chart(data=chartdata, x_label="date",y_label="using_seconds")


# 获取每日总鼠标使用时间数据
response_daily_total = requests.get(f'{backend_url}/api/daily_mouse_usage')
if response_daily_total.status_code == 200:
    daily_total_mouse_usage = response_daily_total.json()['daily_mouse_usage']

    # print(type(daily_total_mouse_usage))
    for i in range(len(daily_total_mouse_usage)):
        daily_total_mouse_usage[i]['total_mouse_time'] = daily_total_mouse_usage[i]['total_mouse_time'] / 60

    # 将数据转换为DataFrame
    df_daily_total = pd.DataFrame(daily_total_mouse_usage)
    df_daily_total['day'] = pd.to_datetime(df_daily_total['day'])

    # Streamlit 应用
    # st.title('Mouse Usage Dashboard')

    # # 显示每日总鼠标使用时间的柱形图
    # st.subheader('Daily Total Mouse Usage')
    st.bar_chart(df_daily_total.set_index('day'), width=100)

else:
    st.error("Failed to fetch data from the backend API")
