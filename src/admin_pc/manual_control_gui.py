import sys
import socket
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import cv2
import numpy as np
from main_gui import MainWindowClass



# UI 파일 로드
manualUi = uic.loadUiType("/home/shim/ui/Arrow keys.ui")[0]
mainUi = uic.loadUiType("/home/lim/dev_ws/deeplearning-repo-2/src/admin_pc/main_gui.ui")[0]



UDP_UI = "127.0.0.1"
UDP_PORT = 7001
BUFFER_SIZE = 65536


class UDPWebcamFrame(QFrame):
    """Custom QFrame to receive and display the UDP video stream."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Queue for thread-safe frame handling
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Start UDP thread to receive video frames
        self.udp_thread = threading.Thread(target=self.receive_frames, daemon=True)
        self.udp_thread.start()

        # Timer to refresh displayed frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update UI every 30ms

        self.latest_frame = None  # Store the latest received frame



    def receive_frames(self):
        """Receives video frames over UDP in a separate thread."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        packet_buffer = {}  # Buffer for reassembling frames

        while True:
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

                # Store packet in buffer
                if frame_id not in packet_buffer:
                    packet_buffer[frame_id] = [None] * total_packets
                packet_buffer[frame_id][packet_num] = packet_data

                # If full frame is received, decode and queue it
                if None not in packet_buffer[frame_id]:
                    full_frame_data = b"".join(packet_buffer[frame_id])
                    frame = cv2.imdecode(np.frombuffer(full_frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    del packet_buffer[frame_id]  # Free the buffer for this frame

            except socket.timeout:
                pass  # Continue if no data received



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



    def closeEvent(self, event):
        """Close UDP thread and timer properly."""
        # Optionally, you can add cleanup for the UDP thread if needed.
        self.timer.stop()
        event.accept()





        
class ManualWindowClass(QMainWindow, manualUi):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Hosbot")

        # 소켓 클라이언트 설정
        self.server_ip = '172.29.146.82'  # 메인 서버의 IP 주소
        self.command_port = 5000  # 명령 전송용 포트
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.command_port))

        # 버튼 클릭 시 명령 전송
        self.pushButton.clicked.connect(self.move_forward)    # Forward
        self.pushButton_3.clicked.connect(self.move_left_side)  # Left Side
        self.pushButton_4.clicked.connect(self.move_right_side)  # Right Side
        self.pushButton_5.clicked.connect(self.move_backward)  # Backward
        self.pushButton_6.clicked.connect(self.stop)  # Stop
        self.pushButton_7.clicked.connect(self.turn_left)  # Left Turn
        self.pushButton_8.clicked.connect(self.turn_right)  # Right Turn

        self.main_btn.clicked.connect(self.open_main_window)

        # Replace the existing QFrame (from UI) with our UDPWebcamFrame
        self.webcam_frame = UDPWebcamFrame(self)
        self.webcam_frame.setGeometry(self.frame.geometry())  # Match the size and position of the UI frame
        self.webcam_frame.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.frame.hide()  # Hide the original placeholder frame



    def open_main_window(self):
        self.manual_control_window = MainWindowClass()  # main_gui.py의 WindowClass 실행
        self.manual_control_window.show()  # main_gui.ui를 실행
        self.close()  # 현재 manaul control 창 닫기



    # 명령 전송 함수
    def send_command(self, command):
        self.client_socket.send(command.encode())



    # 각 버튼에 대한 동작
    def move_forward(self):
        self.send_command('f')  # Forward



    def move_backward(self):
        self.send_command('b')  # Backward



    def move_left_side(self):
        self.send_command('a')  # Left Side



    def move_right_side(self):
        self.send_command('d')  # Right Side



    def stop(self):
        self.send_command('s')  # Stop



    def turn_left(self):
        self.send_command('q')  # Left Turn



    def turn_right(self):
        self.send_command('r')  # Right Turn



    def closeEvent(self, event):
        """Ensure proper cleanup on window close."""
        # Attempt to join the UDP thread with a timeout (if it's still running)
        self.webcam_frame.udp_thread.join(1)
        event.accept()



    # QFrame에 받은 영상 업데이트
    def update_frame(self, frame_data):
        # OpenCV 이미지 포맷을 QImage로 변환
        height, width, channel = frame_data.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # QFrame에 영상 출력
        pixmap = QPixmap.fromImage(q_img)
        self.label_video.setPixmap(pixmap)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = ManualWindowClass()
    myWindows.show()
    sys.exit(app.exec_())