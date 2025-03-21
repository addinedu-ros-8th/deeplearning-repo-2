import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDate, QTimer, Qt
import cv2
from graph_gui import GraphGUI

ui_file = "/path/to/log_gui.ui"  # Load UI file
graph_gui = "/path/to/graph_gui.ui"
Ui_Dialog, _ = uic.loadUiType(ui_file)

class LogGUI(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Set table behavior
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 창 크기에 맞게 확장

        # 📌 초기 최소 행 개수 설정
        self.min_rows = 7  
        self.update_table_rows()

        # Date settings
        today = QDate.currentDate()
        self.start_date.setDate(today)
        self.end_date.setDate(today)
        self.end_date.setEnabled(False)

        self.start_date.dateChanged.connect(self.toggle_end_date)
        self.start_date.dateChanged.connect(self.update_end_date_minimum)

        # Video display settings
        self.label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.video_btn.clicked.connect(self.start_video)  # Connect button click to function

        self.graph_btn.clicked.connect(self.show_graph)

        # Video attributes
        self.cap = None  # Video capture object
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)  # Timer calls update_frame()

    def update_end_date_minimum(self):
        start_date_value = self.start_date.date()
        self.end_date.setMinimumDate(start_date_value)

    def toggle_end_date(self):
        if self.start_date.date() != QDate(1970, 1, 1):
            self.end_date.setEnabled(True)
        else:
            self.end_date.setEnabled(False)

    def start_video(self):
        """Start playing video on QLabel when button is clicked."""
        video_path = "/path/to/people_walk.mp4"  # 🔥 Change this to your video file path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Failed to open video file!")
            return
        
        self.timer.start(30)  # Update frame every 30ms (approx. 33 FPS)

    def update_frame(self):
        """Read frame from video and display it on QLabel."""
        ret, frame = self.cap.read()
        
        if not ret:
            self.timer.stop()  # Stop video when finished
            self.cap.release()
            return

        # Resize frame to match QLabel size
        frame = cv2.resize(frame, (self.label.width(), self.label.height()), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Convert frame to QImage and update QLabel
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.label.setPixmap(pixmap)

    def update_table_rows(self):
        """현재 창 크기에 맞춰 테이블 행 개수 조정"""
        row_height = 30  # 한 행당 높이 (픽셀 단위)
        table_height = self.tableWidget.height()  # 현재 QTableWidget 높이

        # 계산된 행 개수 (최소 7개 이상)
        new_row_count = max(self.min_rows, table_height // row_height)
        self.tableWidget.setRowCount(new_row_count)

    def resizeEvent(self, event):
        """창 크기가 변경될 때 테이블 행 개수 업데이트"""
        self.update_table_rows()
        event.accept()

    def show_graph(self):
        self.graph_window = GraphGUI()
        self.graph_window.show()  # Main.ui를 실행
        self.close()  # 현재 로그인 창 닫기

    def closeEvent(self, event):
        """창 닫을 때 비디오 종료"""
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = LogGUI()
    gui.show()
    sys.exit(app.exec_())
