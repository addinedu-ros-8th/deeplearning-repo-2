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
import speech_recognition as sr

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

# Voice TCP 설정
TCP_IP = '192.168.28.150'
TCP_PORT = 6001

video_save_dir = "/Users/wjsong/dev_ws/deeplearning-repo-2/src/admin_pc/video_out/"

command_dict = {
                    "전진": "FORWARD", 
                    "후진": "BACKWARD",
                    "좌측": "LEFT_MOVE",
                    "우측": "RIGHT_MOVE",
                    "좌회전": "LEFT_TURN",
                    "우회전": "RIGHT_TURN",
                    "정지": "STOP"
                }

REC_IP = "0.0.0.0"
REC_PORT = 6001

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

class CommandSender(QThread):

    def __init__(self):
        super().__init__()
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_sock.settimeout(1.0)
        try:
            self.tcp_sock.connect((TCP_IP, TCP_PORT))
            print(f"TCP 연결 성공: {TCP_IP}:{TCP_PORT}")
        except socket.error as e:
            print(f"TCP 연결 실패: {e}")
            self.tcp_sock = None

        self.running = True
        self.listening = False
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def run(self):
        while self.running:
            with self.mic as source:
                print("음성을 듣는 중...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)

            try:
                text = self.recognizer.recognize_google(audio, language="ko-KR")
                print(f"인식된 단어: {text}")

                if text in command_dict:
                    command = command_dict[text]
                    print(f"명령어 전송: {command}")
                    if self.tcp_sock:
                        self.tcp_sock.send(command.encode('utf-8'))
                    else:
                        print("⚠️ TCP 소켓이 연결되어 있지 않습니다.")

                    print(f"명령어 '{command}' 전송 완료.")
                else:
                    print(f"알 수 없는 명령어: '{text}'")

            except sr.UnknownValueError:
                print("음성을 인식할 수 없습니다.")
            except sr.RequestError as e:
                print(f"음성 인식 서비스 오류 발생: {e}")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.tcp_sock.close()

class RecReceiver(QThread):
    rec_signal = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(1.0)
        self.sock.bind((REC_IP, REC_PORT))
        self.running = True

    def run(self):
        while self.running:
            try:
                data = self.sock.recv(1024).decode()
                action, status = data.split(":")
                
                if action == "REC_ON":
                    self.rec_signal.emit(True, status)
                elif action == "REC_OFF":
                     self.rec_signal.emit(False, status)
                else:
                    print("올바르지 않은 명령어입니다.")
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
    global status
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

        self.record_thread = RecReceiver()
        self.record_thread.rec_signal.connect(self.handle_recording)
        self.record_thread.start()
        self.is_recording = False
        self.frame = None
        self.writer = None
        self.status = None

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

        self.command_thread = CommandSender()

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

    def handle_recording(self, start, rec_status):
        self.status = rec_status
        if start:
            if not self.is_recording and self.frame is not None:
                self.recordingStart()
        else:
            if self.is_recording:
                self.recordingStop()

    def recordingStart(self):
        if self.frame is None:
            print("❌ 녹화 시작 실패: frame이 None입니다.")
            return

        try:
            self.now = str(self.status) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = self.now + ".mp4"
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w, _ = self.frame.shape
            save_path = os.path.join(video_save_dir, file_name)
            self.writer = cv2.VideoWriter(save_path, self.fourcc, 20.0, (w, h))

            if not self.writer.isOpened():
                print(f"❌ VideoWriter 열기 실패: {save_path}")
                self.writer = None
                return

            self.is_recording = True
        except Exception as e:
            print(f"❌ 녹화 시작 중 오류 발생: {e}")

    def recordingStop(self):
        self.is_recording = False
        if self.writer:
            self.writer.release()
            self.writer = None

    def update_frame(self, frame):
        self.frame = frame.copy()
        label_width = self.label.width()
        label_height = self.label.height()
        resized_frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

        if self.is_recording and self.writer:
            try:
                self.writer.write(frame)
            except Exception as e:
                print(f"❌ 프레임 녹화 실패: {e}")


    def manual_mode(self):
        print("Manual Mode Activated")
        self.manual.setChecked(True)
        self.manual_btn.setDisabled(True)
        self.patrol_btn.setDisabled(False)

        if not self.command_thread.isRunning():
            self.command_thread.start()


    def patrol_mode(self):
        print("Patrol Mode Activated")
        if self.command_thread.isRunning():
            self.command_thread.stop()
            self.command_thread = CommandSender()
            
        self.patrol.setChecked(True)
        self.manual_btn.setDisabled(False)
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
        self.record_thread.stop()
        self.command_thread.stop()
        if self.is_recording:
            self.writer.release()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec_())
