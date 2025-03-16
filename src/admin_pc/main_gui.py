import sys
import cv2
import socket
import threading
import numpy as np
import queue
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QHeaderView
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QDate, QTimer, Qt, QThread
from PyQt5 import uic

# Load the UI file
from_class = uic.loadUiType("/path/to/main_gui.ui")[0]

# UDP Video Streaming Configuration
UDP_IP = "local IP"
UDP_PORT = "your port Num"
BUFFER_SIZE = 65536  # Max UDP packet size
ADMIN_CLIENT_IP = 'tcp server IP'
ADMIN_CLIENT_PORT = "your port Num"

# 🚀 TCP 송신용 스레드
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



class UDPWebcamFrame(QFrame):
    """Custom QFrame to receive and display the UDP video stream."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.frame_queue = queue.Queue(maxsize=5)
        self.latest_frame = None  # Store the latest received frame
        self.running = True  # ✅ 스레드 종료 플래그 추가

        # Start UDP thread to receive video frames
        self.udp_thread = threading.Thread(target=self.receive_frames, daemon=True)
        self.udp_thread.start()

        # Timer to refresh displayed frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update UI every 30ms

    def receive_frames(self):
        """Receives video frames over UDP in a separate thread."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        sock.settimeout(1.0)  # ✅ Timeout 추가 (1초) → 빠른 종료 가능

        packet_buffer = {}

        while self.running:  # ✅ UDP 스레드가 종료될 수 있도록 함
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                if data == b"END_OF_FILE":
                    print("[UDP] Video stream ended.")
                    break

                # Extract metadata: "frame_id,packet_num,total_packets||data"
                try:
                    header, packet_data = data.split(b"||", 1)
                    frame_id, packet_num, total_packets = map(int, header.decode().split(","))
                except ValueError:
                    print("[UDP] Corrupt packet received, skipping...")
                    continue

                if frame_id not in packet_buffer:
                    packet_buffer[frame_id] = [None] * total_packets
                packet_buffer[frame_id][packet_num] = packet_data

                if None not in packet_buffer[frame_id]:  # 모든 패킷을 수신했을 때
                    full_frame_data = b"".join(packet_buffer[frame_id])
                    frame = cv2.imdecode(np.frombuffer(full_frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    del packet_buffer[frame_id]  # Free the buffer for this frame
            except socket.timeout:
                continue  # ✅ 1초마다 종료 플래그 확인 가능

        sock.close()  # ✅ 소켓 닫기 (명확한 종료)

    def update_frame(self):
        """Updates the displayed frame from the frame queue."""
        if not self.frame_queue.empty():
            self.latest_frame = self.frame_queue.get()
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        """Override paintEvent to draw the latest received frame."""
        if self.latest_frame is not None:
            painter = QPainter(self)
            h, w, ch = self.latest_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.latest_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()

    def stop_udp_thread(self):
        """✅ UDP 스레드를 안전하게 종료"""
        self.running = False  # 스레드 루프 종료
        try:
            self.sock.close()  # ✅ 소켓 강제 종료
        except Exception as e:
            print(f"[ERROR] Failed to close socket: {e}")
        
        self.udp_thread.join()  # ✅ 스레드가 완전히 종료될 때까지 기다림
        print("[INFO] UDP thread stopped successfully.")

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        today = QDate.currentDate()
        self.start_date.setDate(today)
        self.end_date.setDate(today)
        self.end_date.setEnabled(False)

        self.start_date.dateChanged.connect(self.toggle_end_date)
        self.start_date.dateChanged.connect(self.update_end_date_minimum)

        self.setWindowTitle("Hosbot")

        self.pushButton.clicked.connect(lambda: self.send_command("FORWARD"))
        self.pushButton_4.clicked.connect(lambda: self.send_command("LEFT"))
        self.pushButton_5.clicked.connect(lambda: self.send_command("RIGHT"))
        self.pushButton_6.clicked.connect(lambda: self.send_command("BACKWARD"))
        self.pushButton_7.clicked.connect(lambda: self.send_command("STOP"))
        self.pushButton_8.clicked.connect(lambda: self.send_command("LEFT_TURN"))
        self.pushButton_9.clicked.connect(lambda: self.send_command("RIGHT_TURN"))

        # Replace the existing QFrame (from UI) with our UDPWebcamFrame
        self.webcam_frame = UDPWebcamFrame(self)
        self.webcam_frame.setGeometry(self.frame.geometry())  # Match the size and position of the UI frame
        self.webcam_frame.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.frame.hide()

    def update_end_date_minimum(self):
        start_date_value = self.start_date.date()
        self.end_date.setMinimumDate(start_date_value)

    def toggle_end_date(self):
        if self.start_date.date() != QDate(1970, 1, 1):
            self.end_date.setEnabled(True)
        else:
            self.end_date.setEnabled(False)
    
    def send_command(self, command):
        print(f"Sending command: {command}")
        self.admin_client_thread = AdminClientThread(command)
        self.admin_client_thread.start()

    def closeEvent(self, event):
        """✅ 창을 닫을 때 UDP 스레드와 소켓을 올바르게 정리"""
        print("[INFO] Closing application, stopping UDP thread...")
        self.webcam_frame.stop_udp_thread()  # ✅ UDP 종료
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
