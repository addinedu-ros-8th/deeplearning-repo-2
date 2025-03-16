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

        # UDP 소켓 설정 (영상 수신용)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_port = 5001
        self.udp_socket.bind(('', self.udp_port))

        # 버튼 클릭 시 명령 전송
        self.pushButton.clicked.connect(self.move_forward)    # Forward
        self.pushButton_3.clicked.connect(self.move_left_side)  # Left Side
        self.pushButton_4.clicked.connect(self.move_right_side)  # Right Side
        self.pushButton_5.clicked.connect(self.move_backward)  # Backward
        self.pushButton_6.clicked.connect(self.stop)  # Stop
        self.pushButton_7.clicked.connect(self.turn_left)  # Left Turn
        self.pushButton_8.clicked.connect(self.turn_right)  # Right Turn

        self.main_btn.clicked.connect(self.open_main_window)

        # UDP 영상 수신 스레드
        self.video_thread = VideoReceiverThread(self.udp_socket)
        self.video_thread.new_frame_signal.connect(self.update_frame)
        self.video_thread.start()



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
        self.client_socket.close()  # 종료 시 소켓 닫기
        self.udp_socket.close()  # UDP 소켓 종료



    # QFrame에 받은 영상 업데이트
    def update_frame(self, frame_data):
        # OpenCV 이미지 포맷을 QImage로 변환
        height, width, channel = frame_data.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # QFrame에 영상 출력
        pixmap = QPixmap.fromImage(q_img)
        self.label_video.setPixmap(pixmap)







class VideoReceiverThread(QThread):
    # 새로운 프레임을 받아서 UI로 전달하는 시그널
    new_frame_signal = pyqtSignal(np.ndarray)



    def __init__(self, udp_socket):
        super().__init__()
        self.udp_socket = udp_socket
        self.frame_data = b""
        self.buffer_size = 65536  # UDP 최대 버퍼 크기



    def run(self):
        while True:
            try:
                # UDP 패킷 수신
                data, addr = self.udp_socket.recvfrom(self.buffer_size)
                # 패킷을 받아서 하나의 프레임을 완성
                self.frame_data += data

                # 프레임이 완성되면
                if b'END' in self.frame_data:
                    # 프레임의 끝을 찾았을 때 프레임 분리
                    frame = self.frame_data.split(b'END')[0]
                    self.frame_data = self.frame_data.split(b'END')[1]  # 나머지 데이터
                    # 프레임을 OpenCV 이미지로 변환
                    np_frame = np.frombuffer(frame, dtype=np.uint8)
                    image = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
                    # 새로운 프레임이 오면 시그널로 전달
                    self.new_frame_signal.emit(image)

            except Exception as e:
                print(f"UDP 오류: {e}")
                break






if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = ManualWindowClass()
    myWindows.show()
    sys.exit(app.exec_())