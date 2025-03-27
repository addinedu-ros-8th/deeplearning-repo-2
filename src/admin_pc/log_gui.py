import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDate, QTimer, Qt
import mysql.connector
import cv2
from graph_gui import GraphGUI
from dotenv import load_dotenv
import os

load_dotenv()

LOG_GUI = os.environ.get("PATH_TO_LOG_GUI")
GRAPH_GUI = os.environ.get("PATH_TO_GRAPH_GUI")
MAIN_GUI = os.environ.get("PATH_TO_MAIN_GUI")

HOST = os.environ.get("MYSQL_HOST")
USER = os.environ.get("MYSQL_USER")
PASSWD = os.environ.get("MYSQL_PASSWD")
DB_NAME = os.environ.get("DB_NAME")

ui_file = LOG_GUI
graph_gui = GRAPH_GUI
main_gui = MAIN_GUI

VIDEO = os.environ.get("VIDEO_PATH")

Ui_Dialog, _ = uic.loadUiType(ui_file)

class LogGUI(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.remote = mysql.connector.connect(
            host=HOST,
            user=USER,
            password=PASSWD,
            database=DB_NAME
        )

        # Set table behavior
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 창 크기에 맞게 확장

        # 📌 초기 최소 행 개수 설정
        self.min_rows = 7  
        self.update_table_rows()

        # __init__ 끝부분
        self.load_data_from_db()  # 👉 DB에서 데이터를 불러오고 테이블에 채움


        # Date settings
        default_date = QDate(2000, 1, 1)  # 의미 없는 날짜
        self.start_date.setDate(default_date)
        self.end_date.setDate(default_date)

        self.start_date.setSpecialValueText("--/--/--")
        self.end_date.setSpecialValueText("--/--/--")

        self.start_date.setMinimumDate(default_date)
        self.end_date.setMinimumDate(default_date)

        self.end_date.setEnabled(False)

        # Connect
        self.start_date.dateChanged.connect(self.toggle_end_date)
        self.start_date.dateChanged.connect(self.update_end_date_minimum)
        self.end_date.dateChanged.connect(self.load_data_from_db)



        self.tableWidget.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tableWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.end_date.dateChanged.connect(self.load_data_from_db)

        # Video display settings
        self.label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.video_btn.clicked.connect(self.start_video)  # Connect button click to function

        self.graph_btn.clicked.connect(self.show_graph)
        self.main_btn.clicked.connect(self.back_to_main)

        # Video attributes
        self.cap = None  # Video capture object
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)  # Timer calls update_frame()

    def update_end_date_minimum(self):
        start_date_value = self.start_date.date()
        self.end_date.setMinimumDate(start_date_value)

    def toggle_end_date(self):
        # 기본값이 2000년 1월 1일이면 선택 안 한 걸로 간주
        if self.start_date.date() != QDate(2000, 1, 1):
            self.end_date.setEnabled(True)
        else:
            self.end_date.setEnabled(False)


    def start_video(self):
        selected_row = self.tableWidget.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a row first.")
            return

        timestamp_item = self.tableWidget.item(selected_row, 3)  # 4번째 열 (TIMESTAMP)
        if not timestamp_item:
            QMessageBox.warning(self, "Missing Data", "Timestamp not found in selected row.")
            return

        raw_timestamp = timestamp_item.text()  # ex: "2025-03-26 15:42:10"

        try:
            from datetime import datetime
            dt = datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S")
            video_filename = dt.strftime("%y%m%d_%H%M%S") + ".mp4"
        except ValueError:
            QMessageBox.warning(self, "Format Error", f"Invalid timestamp format: {raw_timestamp}")
            return

        video_path = os.path.join(VIDEO, video_filename)

        if not os.path.exists(video_path):
            QMessageBox.warning(self, "File Not Found", f"No video found for: {video_filename}")
            return

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Failed to open video file!")
            return

        self.timer.start(30)


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
        table_height = self.tableWidget.height()
        
        self.row_count = max(self.min_rows, table_height // row_height)  # 👉 저장해둠
        self.tableWidget.setRowCount(self.row_count)


    def resizeEvent(self, event):
        """창 크기가 변경될 때 테이블 행 개수 업데이트"""
        self.update_table_rows()
        event.accept()
    
    def load_data_from_db(self):
        """두 테이블에서 데이터를 불러와 시간 순으로 정렬 후 테이블에 표시 (날짜 필터 포함)"""
        cursor = self.remote.cursor()

        try:
            limit = getattr(self, "row_count", 10)

            # 날짜 가져오기
            start = self.start_date.date().toString("yyyy-MM-dd")
            end = self.end_date.date().toString("yyyy-MM-dd")

            query = f"""
            SELECT 
                alog.alogId AS logId,
                r.modelName AS robotName,
                alog.placeId AS location,
                a.actionName AS logName,
                alog.createDate AS timeStamp
            FROM Action_Log alog
            JOIN Robot r ON alog.robotId = r.robotId
            JOIN Action a ON alog.actionId = a.actionId
            WHERE alog.createDate BETWEEN '{start}' AND '{end}'

            UNION ALL

            SELECT 
                elog.elogId AS logId,
                r.modelName AS robotName,
                elog.placeId AS location,
                e.eventName AS logName,
                elog.createDate AS timeStamp
            FROM Event_Log elog
            JOIN Robot r ON elog.robotId = r.robotId
            JOIN Event e ON elog.eventId = e.eventId
            WHERE elog.createDate BETWEEN '{start}' AND '{end}'

            ORDER BY timeStamp DESC
            LIMIT {limit}
            """
            cursor.execute(query)
            results = cursor.fetchall()

            self.tableWidget.setRowCount(len(results))
            self.tableWidget.setColumnCount(4)

            headers = ["ROBOT NAME", "LOCATION", "LOG NAME", "TIMESTAMP"]
            self.tableWidget.setHorizontalHeaderLabels(headers)

            for row_idx, row_data in enumerate(results):
                # row_data = (logId, robotName, location, logName, timeStamp)
                visible_data = row_data[1:]  # logId 제외하고 뒤 4개만 사용
                for col_idx, value in enumerate(visible_data):
                    item = QTableWidgetItem("" if value is None else str(value))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.tableWidget.setItem(row_idx, col_idx, item)


        except mysql.connector.Error as e:
            QMessageBox.critical(self, "DB Error", f"Failed to load data: {e}")
        finally:
            cursor.close()





    def show_graph(self):
        self.graph_window = GraphGUI()
        self.graph_window.show()  # Main.ui를 실행
        self.close()  # 현재 로그인 창 닫기

    # Change the method like this:
    def back_to_main(self):
        from main_gui import MainGUI  # 👈 import here
        self.main_window = MainGUI()
        self.main_window.show()
        self.close()


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
