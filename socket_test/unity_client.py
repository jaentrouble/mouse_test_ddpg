import socket
import time
import numpy as np
import json
from matplotlib import pyplot as plt
import imageio

host = 'localhost'
port = 7777
count = 0

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
            data = conn.recv(51200)
            if not data:
                break
            image = np.frombuffer(data[:25600], dtype=np.uint8)
            info_raw = data[25600:].decode('utf-8')
            info = json.loads(info_raw)
            message = json.dumps({
                'move' : np.random.random()*0.1,
                'turn' : np.random.random(),
            })
            conn.sendall(message.encode('utf-8'))

image = image.reshape((80,80,4))[::-1,...,:3]
imageio.imwrite('test.png',image)

print(f'{count/(time.time()-start_time)} frames/sec')