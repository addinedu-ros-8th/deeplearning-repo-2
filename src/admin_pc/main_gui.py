import sys
import socket
import cv2
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from log_gui import LogGUI
import mysql.connector
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

MAIN_GUI = os.environ.get("PATH_TO_MAIN_GUI")
LOG_GUI = os.environ.get("PATH_TO_LOG_GUI")

# UDP 설정
UDP_IP = os.environ.get("ADMIN_IP")
UDP_PORT = int(os.environ.get("MAIN_PORT"))
BUFFER_SIZE = 65536

HOST = os.environ.get("MYSQL_HOST")
USER = os.environ.get("MYSQL_USER")
PASSWD = os.environ.get("MYSQL_PASSWD")
DB_NAME = os.environ.get("DB_NAME")

# UI 파일 로드
ui_file = MAIN_GUI
Ui_Dialog, _ = uic.loadUiType(ui_file)


class VideoReceiver(QThread):
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(1.0)
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
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receiver error: {e}")
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
        self.remote = mysql.connector.connect(
            host=HOST,
            user=USER,
            password=PASSWD,
            database=DB_NAME
        )

        self.waiting.setChecked(True)
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

        self.label_update_timer = QTimer(self)
        self.label_update_timer.timeout.connect(self.update_reservation_label)
        self.label_update_timer.start(1000)

        self.reservation_timer = QTimer(self)
        self.reservation_timer.timeout.connect(self.check_reservation)
        self.reservation_timer.start(1000)

        self.last_triggered_time = None  # ⛑️ 마지막으로 실행된 예약 시간


    def check_reservation(self):
        try:
            now = datetime.now()
            now_time_str = now.strftime("%H:%M:%S")

            cursor = self.remote.cursor()
            cursor.execute(
                "SELECT reservationTime FROM Reservation WHERE reservationTime > %s ORDER BY reservationTime ASC LIMIT 1",
                (now_time_str,)
            )
            result = cursor.fetchone()
            cursor.close()

            if result:
                t = result[0]

                if isinstance(t, timedelta):
                    total_seconds = int(t.total_seconds())
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    reserved_datetime = now.replace(hour=hours, minute=minutes, second=seconds, microsecond=0)
                else:
                    reserved_datetime = datetime.combine(now.date(), t)

                # ✅ 중복 실행 방지
                if self.last_triggered_time == reserved_datetime:
                    return  # 이미 실행된 예약이므로 무시

                if abs((reserved_datetime - now).total_seconds()) < 1:
                    print("⏰ 예약 시간 도달! → 자동 Patrol Mode 전환")
                    self.patrol_mode()
                    self.last_triggered_time = reserved_datetime  # ⛑️ 예약 실행 시간 저장
                else:
                    print("예약 시간에 도달하지 않았습니다.")
            else:
                print("⛔️ 오늘 남은 예약 없음.")
        except mysql.connector.Error as e:
            print(f"[DB ERROR] 예약 확인 실패: {e}")


    def update_reservation_label(self):
        try:
            now_time_str = datetime.now().strftime("%H:%M:%S")
            cursor = self.remote.cursor()
            cursor.execute(
                "SELECT reservationTime FROM Reservation WHERE reservationTime > %s ORDER BY reservationTime ASC LIMIT 1",
                (now_time_str,)
            )
            result = cursor.fetchone()
            cursor.close()

            if result:
                t = result[0]
                if isinstance(t, timedelta):
                    total_seconds = int(t.total_seconds())
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                else:
                    hours = t.hour
                    minutes = t.minute
                    seconds = t.second

                formatted = f"{hours:02d}시 {minutes:02d}분 {seconds:02d}초"
                self.res_label.setText(f"Next Reservation : {formatted}")
            else:
                self.res_label.setText("Next Reservation : 없음")
        except mysql.connector.Error as e:
            self.res_label.setText("Next Reservation : DB 에러")
            print(f"[DB ERROR] 예약 시간 표시 실패: {e}")

    def update_frame(self, frame):
        label_width = self.label.width()
        label_height = self.label.height()
        frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

    def manual_mode(self):
        print("Manual Mode Activated")
        self.manual.setChecked(True)
        self.manual_btn.setDisabled(True)
        self.patrol_btn.setDisabled(True)
        QTimer.singleShot(3000, self.enable_mode_buttons)

    def patrol_mode(self):
        print("Patrol Mode Activated")
        self.patrol.setChecked(True)
        self.manual_btn.setDisabled(True)
        self.patrol_btn.setDisabled(True)
        QTimer.singleShot(3000, self.enable_mode_buttons)

    def enable_mode_buttons(self):
        self.waiting.setChecked(True)
        self.manual_btn.setEnabled(True)
        self.patrol_btn.setEnabled(True)

    def restart_video_thread_and_show(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
        self.video_thread = VideoReceiver()
        self.video_thread.frame_received.connect(self.update_frame)
        self.video_thread.start()
        self.show()

    def show_log(self):
        self.log_window = LogGUI()
        self.log_window.show()
        self.log_window.destroyed.connect(self.restart_video_thread_and_show)
        self.hide()

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec_())
