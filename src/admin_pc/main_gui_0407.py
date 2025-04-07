import sys
import socket
import cv2
import numpy as np
import subprocess
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel
from log_gui import LogGUI

# UDP 설정
UDP_IP = "0.0.0.0"
UDP_PORT = 5001
BUFFER_SIZE = 65536

# TCP 설정 (라즈베리파이 주소)
TCP_IP = "172.24.125.150"
TCP_PORT = 6001
TCP_SOCKET = None

# UI 파일 경로
ui_file = "/home/shim/dev_ws/gui/main_gui.ui"
log_gui = "/home/shim/dev_ws/gui/log_gui.ui"
Ui_Dialog, _ = uic.loadUiType(ui_file)

KEY_MAPPING = {
    "W": "FORWARD",
    "A": "LEFT_MOVE",
    "D": "RIGHT_MOVE",
    "S": "STOP",
    "Q": "LEFT_TURN",
    "E": "RIGHT_TURN",
    "X": "BACKWARD"
}


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
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
                frame = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.frame_received.emit(frame)
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
        self.setupUi(self)

        self.waiting.setChecked(True)
        self.waiting.setFocusPolicy(Qt.NoFocus)
        self.patrol.setFocusPolicy(Qt.NoFocus)
        self.manual.setFocusPolicy(Qt.NoFocus)
        self.manual_btn.setFocusPolicy(Qt.NoFocus)
        self.patrol_btn.setFocusPolicy(Qt.NoFocus)
        self.log_btn.setFocusPolicy(Qt.NoFocus)

        self.waiting.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.patrol.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.manual.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.label.setStyleSheet("background-color: black;")

        self.video_thread = VideoReceiver()
        self.video_thread.frame_received.connect(self.update_frame)
        self.video_thread.start()

        self.manual_btn.clicked.connect(self.manual_mode)
        self.patrol_btn.clicked.connect(self.patrol_mode)
        self.log_btn.clicked.connect(self.show_log)

        self.label.setFocusPolicy(Qt.StrongFocus)
        self.label.setFocus()

        self.connect_tcp()

    def connect_tcp(self):
        """TCP 서버와 연결"""
        global TCP_SOCKET
        try:
            if TCP_SOCKET is not None:
                TCP_SOCKET.close()
            TCP_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            TCP_SOCKET.connect((TCP_IP, TCP_PORT))
            print(f"Connected to {TCP_IP} on port {TCP_PORT}")
        except Exception as e:
            print(f"Error connecting to TCP server: {e}")
            TCP_SOCKET = None

    def update_frame(self, frame):
        """프레임을 표시"""
        label_width = self.label.width()
        label_height = self.label.height()

        frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        """키보드 이벤트 처리"""
        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            print("Exiting mode, switching to waiting mode.")
            self.waiting.setChecked(True)
            self.manual.setChecked(False)
            self.patrol.setChecked(False)

            self.manual_btn.setEnabled(True)
            self.patrol_btn.setEnabled(True)
            self.log_btn.setEnabled(True)

            self.send_stop_yolo()
            self.send_key_to_raspberry("STOP")  # 수동 정지 명령 추가
            return

        if self.patrol.isChecked():
            return

        if self.manual.isChecked():
            key_name = QtGui.QKeySequence(key).toString()
            self.process_key(key_name)

    def process_key(self, key_name):
        """입력 받은 키를 처리"""
        print(f"Key pressed: {key_name}")
        command = KEY_MAPPING.get(key_name.upper(), key_name)
        self.label_3.setText(f"Pressed: {command}")
        self.send_key_to_raspberry(command)

    def send_key_to_raspberry(self, key_name):
        """라즈베리파이에 명령 전송"""
        global TCP_SOCKET
        try:
            if TCP_SOCKET is None:
                self.connect_tcp()
            TCP_SOCKET.sendall((key_name + "\n").encode('utf-8'))
        except Exception as e:
            print(f"Error sending key to Raspberry Pi: {e}")
            self.connect_tcp()

    def manual_mode(self):
        """수동 모드"""
        print("Manual Mode Activated")
        self.manual.setChecked(True)
        self.patrol.setChecked(False)
        self.waiting.setChecked(False)

        self.manual_btn.setEnabled(False)
        self.patrol_btn.setEnabled(False)
        self.log_btn.setEnabled(True)

    def patrol_mode(self):
        """순찰 모드"""
        print("Patrol Mode Activated")
        self.patrol.setChecked(True)
        self.manual.setChecked(False)
        self.waiting.setChecked(False)

        self.patrol_btn.setEnabled(False)
        self.manual_btn.setEnabled(False)
        self.log_btn.setEnabled(True)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('127.0.0.1', 9000))
                s.sendall(b'START_YOLO')
                response = s.recv(1024).decode()
                print(f"서버 응답: {response}")
        except Exception as e:
            print(f"서버 연결 실패: {e}")

    def send_stop_yolo(self):
        """YOLO 종료 명령 전송"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('127.0.0.1', 9000))
                s.sendall(b'STOP_YOLO')
                response = s.recv(1024).decode()
                print(f"YOLO 종료 응답: {response}")
        except Exception as e:
            print(f"YOLO 종료 실패: {e}")

    def show_log(self):
        """로그 창 열기"""
        self.log_window = LogGUI()
        self.log_window.show()
        self.hide()

    def closeEvent(self, event):
        """닫기 이벤트 처리"""
        self.video_thread.stop()
        event.accept()

    def showEvent(self, event):
        """화면 표시 이벤트 처리"""
        self.label.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec_())
