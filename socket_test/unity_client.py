import socket
import time
import numpy as np
import json
from matplotlib import pyplot as plt
import cv2

host = 'localhost'
port = 7777
count = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('unitytest.mp4',fourcc,60, (80,80))

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    print('listening')
    s.listen()
    conn, addr = s.accept()
    print('connected')
    start_time = time.time()
    with conn:
        while True:
            count +=1
            try:
                data = conn.recv(51200)
            except ConnectionResetError:
                break
            if not data:
                break
            image = np.frombuffer(data[:25600], dtype=np.uint8)
            info_raw = data[25600:].decode('utf-8')
            info = json.loads(info_raw)

            frame = image.reshape((80,80,4))[::-1,...,2::-1]
            writer.write(frame)

            message = json.dumps({
                'move' : np.random.random()*0.1,
                'turn' : np.random.random()-0.5,
            })
            conn.sendall(message.encode('utf-8'))

writer.release()

print(f'{count/(time.time()-start_time)} frames/sec')