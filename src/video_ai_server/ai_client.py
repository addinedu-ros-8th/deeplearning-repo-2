import sys
import socket
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

# UDP 클라이언트
class VideoReceiver(QThread):
    frame_received = pyqtSignal(np.ndarray)  # 프레임 수신 시 메인 스레드로 전달

    def __init__(self, client_ip, client_port, server_ip, server_port):
        super().__init__()
        self.client_ip = client_ip
        self.client_port = client_port
        self.server_ip = server_ip
        self.server_port = server_port
        self.running = True

    def run(self):
        # 비디오 수신용 소켓
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.client_ip, self.client_port))
        print("Client is receiving video...")

        # 서버로 데이터를 전송할 소켓
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        while self.running:
            data, addr = sock.recvfrom(65507)  # 최대 UDP 패킷 크기
            nparr = np.frombuffer(data, np.uint8)  # 바이트 데이터를 numpy 배열로 변환
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 이미지로 디코딩

            if frame is not None:
                print("Video frame received")  # 비디오 프레임 수신 메시지

                # 프레임을 서버로 전송
                server_sock.sendto(data, (self.server_ip, self.server_port))  # 서버 IP, 포트로 전송

        sock.close()
        server_sock.close()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# 애플리케이션 실행
if __name__ == "__main__":
    video_receiver = VideoReceiver('127.0.0.1', 7000, '172.29.146.177', 5001)  # UDP 클라이언트 IP와 포트, 서버 IP와 포트 설정
    video_receiver.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nClient stopped.")
        video_receiver.stop()
