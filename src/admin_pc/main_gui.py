import sys
import socket
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QLabel
from log_gui import LogGUI
from dotenv import load_dotenv
import os

load_dotenv()

MAIN_GUI = os.environ.get("PATH_TO_MAIN_GUI")
LOG_GUI = os.environ.get("PATH_TO_LOG_GUI")


IP = os.environ.get("IP")
PORT = os.environ.get("SERVER_PORT")

# UDP 설정
UDP_IP = IP
UDP_PORT = PORT
BUFFER_SIZE = 65536  # UDP 패킷 크기

# UI 파일 로드
ui_file = MAIN_GUI
log_gui = LOG_GUI
Ui_Dialog, _ = uic.loadUiType(ui_file)

class VideoReceiver(QThread):
    """UDP로 영상을 수신하는 스레드"""
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.running = True

    def run(self):
        while self.running:
            try:
                # UDP 데이터 수신
                data, _ = self.sock.recvfrom(BUFFER_SIZE)

                # NumPy 배열 변환
                frame = np.frombuffer(data, dtype=np.uint8)

                # OpenCV로 디코딩 (JPEG 포맷 가정)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                if frame is not None:
                    self.frame_received.emit(frame)  # GUI로 전송
            except Exception as e:
                print(f"Error: {e}")
                break

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.sock.close()

class MainGUI(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 기존 UI 로드

        self.waiting.setChecked(True)

        # 마우스만 투명하게 만들기 (비활성화 느낌 없음)
        self.waiting.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.patrol.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.manual.setAttribute(Qt.WA_TransparentForMouseEvents)


        # 🔥 UI에서 만든 QLabel을 사용해야 함 (새로 만들 필요 없음)
        # 기존 self.video_label = QLabel(self) 제거하고, UI에 있는 QLabel을 그대로 사용
        self.label.setStyleSheet("background-color: black;")  # 배경색 유지

        # UDP 영상 수신 스레드 시작
        self.video_thread = VideoReceiver()
        self.video_thread.frame_received.connect(self.update_frame)
        self.video_thread.start()

        # 기존 버튼 기능 유지
        self.manual_btn.clicked.connect(self.manual_mode)
        self.patrol_btn.clicked.connect(self.patrol_mode)
        self.log_btn.clicked.connect(self.show_log)

    def update_frame(self, frame):
        """UDP에서 받은 프레임을 QLabel 크기에 맞게 조정하여 표시"""
        
        # QLabel 크기 가져오기
        label_width = self.label.width()
        label_height = self.label.height()

        # 프레임을 QLabel 크기에 맞게 리사이징
        frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # QImage 변환
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # QLabel 업데이트
        self.label.setPixmap(pixmap)



    def manual_mode(self):
        print("Manual Mode Activated")
        self.manual.setChecked(True)
        
        # 버튼 비활성화
        self.manual_btn.setDisabled(True)
        self.patrol_btn.setDisabled(True)

        # 3초 후에 다시 활성화
        QTimer.singleShot(3000, self.enable_mode_buttons)


    def patrol_mode(self):
        print("Patrol Mode Activated")
        self.patrol.setChecked(True)
        
        # 버튼 비활성화
        self.manual_btn.setDisabled(True)
        self.patrol_btn.setDisabled(True)

        # 3초 후에 다시 활성화
        QTimer.singleShot(3000, self.enable_mode_buttons)

    def enable_mode_buttons(self):
        self.waiting.setChecked(True)
        self.manual_btn.setEnabled(True)
        self.patrol_btn.setEnabled(True)



    def show_log(self):
        """로그 창 열기 (MainGUI를 숨겼다가 LogGUI가 닫힐 때 다시 표시)"""
        self.log_window = LogGUI()
        self.log_window.show()
        self.hide()  # 🔥 MainGUI 창 숨기기

    def closeEvent(self, event):
        """창 닫을 때 스레드 종료"""
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec_())
