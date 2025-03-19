import sys
import socket
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5 import uic

MAX_UDP_SIZE = 65507
SERVER_IP = '0.0.0.0'
SERVER_PORT = 5001
ADMIN_CLIENT_IP = '127.0.0.1'  # Admin Client (7001)
ADMIN_CLIENT_PORT = 7001
ui_file_path = "/home/shim/ui/main_gui.ui"
form_class, base_class = uic.loadUiType(ui_file_path)

# TCP 송신용 스레드
class AdminClientThread(QThread):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((ADMIN_CLIENT_IP, ADMIN_CLIENT_PORT))
                sock.sendall(self.message.encode())
        except Exception as e:
            print(f"Error sending to Admin Client: {e}")

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
        self.admin_client_thread = AdminClientThread(command)
        self.admin_client_thread.start()

    def closeEvent(self, event):
        self.video_receiver_thread.quit()
        self.video_receiver_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
