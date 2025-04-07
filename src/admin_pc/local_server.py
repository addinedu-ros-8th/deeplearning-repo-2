# yolo_server.py
import socket
import subprocess
import signal
import os

HOST = '127.0.0.1'
PORT = 9000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("YOLO Server: Waiting for trigger...")

yolo_process = None  # 전역 변수로 프로세스 추적

while True:
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    data = conn.recv(1024).decode().strip()

    if data == "START_YOLO":
        if yolo_process is None or yolo_process.poll() is not None:
            print(">> YOLO 실행")
            yolo_process = subprocess.Popen(["python3", "/home/shim/dev_ws/gui/sign_direction4.py"])
            conn.sendall("YOLO_STARTED".encode())
        else:
            conn.sendall("YOLO_ALREADY_RUNNING".encode())

    elif data == "STOP_YOLO":
        if yolo_process and yolo_process.poll() is None:
            print(">> YOLO 종료 중...")
            yolo_process.terminate()
            yolo_process.wait()
            yolo_process = None
            conn.sendall("YOLO_STOPPED".encode())
        else:
            conn.sendall("YOLO_NOT_RUNNING".encode())

    else:
        conn.sendall("UNKNOWN_COMMAND".encode())

    conn.close()

