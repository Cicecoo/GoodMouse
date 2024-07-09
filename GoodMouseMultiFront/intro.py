import time
import streamlit as st


st.title("GoodMouse - 办公手部健康助手")

intro = [
" 无 论 是 在 办 公 室 、 还 是 居 家 办 公 , 往 往 离 不 开 电 脑 , 长 期 握 鼠 标 的 人 们 常 为 腱 鞘 炎 这 一 “ 顽 疾 ” 所 困 扰 。",
" 实 际 上 , 腱 鞘 炎 只 需 适 时 放 松 双 手 就 能 轻 松 预 防 、 从 腱 鞘 炎 中 恢 复 也 需 要 避 免 过 度 用 手 , 但 人 们 往 往 握 住 鼠 标 就 “ 忘 了 时 间 ” , 回 过 神 来 手 部 已 经 开 始 疼 痛 不 适 。",
" 针 对 上 述 问 题 , 我 们 提 出 了 本 作 品 , 综 合 鼠 标 操 作 习 惯 纠 正 、 手 部 运 动 检 测 等 , 助 力 办 公 健 康 。",
" 基 于 关 键 点 检 测 技 术 , 实 现 久 握 鼠 标 提 醒 及 久 转 滚 轮 提 醒 , 同 时 记 录 用 手 数 据 , 通 过 a p p 向 用 户 提 供 手 部 健 康 报 告 ; ",
" 结 合 网 络 通 信 功 能 , 通 过 手 势 控 制 等 提 供 暂 时 替 代 鼠 标 的 方 案 , 放 松 操 作 两 不 误 ; " ,
" 此 外 , 提 供 手 势 操 指 导 功 能 , 帮 助 用 户 更 有 效 地 活 动 双 手 、 增 添 乐 趣 。",
]

def line1():
    for i in intro[0].split(' '):
        yield i
        time.sleep(0.05)

def line2():
    for i in intro[1].split(' '):
        yield i
        time.sleep(0.05)


def line3():
    for i in intro[2].split(' '):
        yield i
        time.sleep(0.05)


def line4():
    for i in intro[3].split(' '):
        yield i
        time.sleep(0.05)


def line5():
    for i in intro[4].split(' '):
        yield i
        time.sleep(0.05)

def line6():
    for i in intro[5].split(' '):
        yield i
        time.sleep(0.05)

time.sleep(1.5)
st.write_stream(line1)
time.sleep(1)
st.write_stream(line2)
time.sleep(1)
st.write_stream(line3)
time.sleep(1)
st.write_stream(line4)
time.sleep(1)
st.write_stream(line5)
time.sleep(1)
st.write_stream(line6)
