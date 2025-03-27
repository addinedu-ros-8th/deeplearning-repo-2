import sys
import socket
import threading
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5 import uic

# ==============================
# Server Configuration
# ==============================
MAX_UDP_SIZE = 65507
SERVER_IP = '0.0.0.0'  # 서버가 모든 인터페이스에서 수신 대기
SERVER_PORT = 5000
MAIN_SERVER_IP = "192.168.65.206"  # 메인 서버 IP 주소
MAIN_SERVER_PORT = 5001
ui_file_path = "/home/shim/ui/main_gui.ui"
form_class, base_class = uic.loadUiType(ui_file_path)

# ==============================
# TCP 송신용 스레드
# ==============================
class AdminClientThread(QThread):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((MAIN_SERVER_IP, MAIN_SERVER_PORT))  # Admin 클라이언트 대신 메인 서버로 전송
                sock.sendall(self.message.encode())
        except Exception as e:
            print(f"Error sending to Main Server: {e}")

# ==============================
# Video Frame 클래스 (영상 표시용)
# ==============================
class VideoFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.image = None

    def set_image(self, img):
        self.image = img
        self.update()

    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            pixmap = QPixmap.fromImage(self.image)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()

# ==============================
# Video Receiver Thread (UDP로 영상 수신)
# ==============================
class VideoReceiverThread(QThread):
    new_frame_signal = pyqtSignal(QImage)

    def __init__(self, server_ip, server_port):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.packet_buffer = b""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.server_ip, self.server_port))

    def run(self):
        while True:
            try:
                packet, addr = self.sock.recvfrom(MAX_UDP_SIZE)
                self.packet_buffer += packet

                if len(self.packet_buffer) > 4 and self.packet_buffer[:4] == b'\xff\xd8\xff\xe0':
                    nparr = np.frombuffer(self.packet_buffer, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        height, width, channels = frame.shape
                        bytes_per_line = channels * width
                        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                        self.new_frame_signal.emit(qimg)

                    self.packet_buffer = b""

            except Exception as e:
                print(f"Error receiving data: {e}")

# ==============================
# Admin 명령을 메인 서버로 전송하는 함수
# ==============================
def send_to_main_server(message):
    """메인 서버로 명령 전송"""
    try:
        main_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        main_sock.connect((MAIN_SERVER_IP, MAIN_SERVER_PORT))
        main_sock.sendall(message.encode("utf-8"))
        main_sock.close()
        print(f"[→] Sent to Main Server: {message}")
    except Exception as e:
        print(f"[!] Failed to send to main server: {e}")

# ==============================
# Admin GUI에서 명령을 받아 메인 서버로 전달하는 스레드
# ==============================
def handle_admin_commands():
    """Admin GUI에서 명령을 받아 메인 서버로 전달"""
    admin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    admin_sock.bind(("0.0.0.0", MAIN_SERVER_PORT))  # Admin 명령을 메인 서버로 전송
    admin_sock.listen(5)

    print(f"[*] Listening for admin commands on port {MAIN_SERVER_PORT}...")

    while True:
        client, addr = admin_sock.accept()
        print(f"[*] Connection from {addr}")

        try:
            while True:
                data = client.recv(1024).decode("utf-8")
                if not data:
                    break
                
                print(f"[Admin GUI] Received: {data}")

                # 메인 서버로 전달
                send_to_main_server(data)
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            client.close()

# ==============================
# PyQt5 GUI 클래스
# ==============================
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.frame = self.findChild(QFrame, "frame")
        self.video_frame = VideoFrame()
        self.layout = QVBoxLayout(self.frame)
        self.layout.addWidget(self.video_frame)

        self.video_receiver_thread = VideoReceiverThread(SERVER_IP, SERVER_PORT)
        self.video_receiver_thread.new_frame_signal.connect(self.update_video_frame)
        self.video_receiver_thread.start()

        self.pushButton_1.clicked.connect(lambda: self.send_command("FORWARD"))
        self.pushButton_2.clicked.connect(lambda: self.send_command("BACKWARD"))
        self.pushButton_3.clicked.connect(lambda: self.send_command("LEFT_TURN"))
        self.pushButton_4.clicked.connect(lambda: self.send_command("RIGHT_TURN"))
        self.pushButton_5.clicked.connect(lambda: self.send_command("LEFT_MOVE"))
        self.pushButton_6.clicked.connect(lambda: self.send_command("RIGHT_MOVE"))
        self.pushButton_7.clicked.connect(lambda: self.send_command("STOP"))
        
    def update_video_frame(self, qimg):
        self.video_frame.set_image(qimg)

    def send_command(self, command):
        print(f"Sending command: {command}")
        send_to_main_server(command)

    def closeEvent(self, event):
        self.video_receiver_thread.quit()
        self.video_receiver_thread.wait()
        event.accept()

if __name__ == "__main__":
    # Admin 명령 처리 스레드 시작
    threading.Thread(target=handle_admin_commands, daemon=True).start()

    # PyQt5 GUI 실행
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
