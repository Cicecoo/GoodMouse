import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.43.229', 2238))
while 1:
    msg = 'mouse_control_info'
    client.send(msg.encode('utf-8'))