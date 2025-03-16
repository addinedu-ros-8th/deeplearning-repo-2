import sys
import cv2
import socket
import threading
import numpy as np
import queue
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QPushButton, QHeaderView
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QDate, QThread, pyqtSignal
from PyQt5 import uic
from udp_receiver import WebcamFrame




ADMIN_CLIENT_IP = 'local IP'  # Admin Client (7001)
ADMIN_CLIENT_PORT = "your port Num"




# Load the UI file
mainUi = uic.loadUiType("/path/to/main_gui.ui")[0]




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




class MainWindowClass(QMainWindow, mainUi):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Hosbot")
        
        # Set table header to stretch its columns
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Set today’s date on start_date and end_date
        today = QDate.currentDate()
        self.start_date.setDate(today)
        self.end_date.setDate(today)
        self.end_date.setEnabled(False)

        # Connect date changed events
        self.start_date.dateChanged.connect(self.toggle_end_date)
        self.start_date.dateChanged.connect(self.update_end_date_minimum)
        self.manualcontrol_btn.clicked.connect(self.open_manual_control_window)
        self.manualcontrol_btn.clicked.connect(self.close)

        # Replace the existing QFrame (from UI) with our UDPWebcamFrame
        self.webcam_frame = WebcamFrame(self)
        self.webcam_frame.setGeometry(self.frame.geometry())  # Match the size and position of the UI frame
        self.webcam_frame.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.frame.hide()  # Hide the original placeholder frame



    def update_end_date_minimum(self):
        start_date_value = self.start_date.date()
        self.end_date.setMinimumDate(start_date_value)



    def toggle_end_date(self):
        if self.start_date.date() != QDate(1970, 1, 1):
            self.end_date.setEnabled(True)
        else:
            self.end_date.setEnabled(False)


    
    def update_video_frame(self, qimg):
        self.video_frame.set_image(qimg)



    def closeEvent(self, event):
        # """ Close UDP thread properly when exiting """
        # self.webcam_frame.udp_thread.join(1)  # Stop UDP thread safely
        event.accept()



    def open_manual_control_window(self):
        from manual_control_gui import WindowClass
        self.manual_gui = QtWidgets.QMainWindow()
        self.ui = WindowClass()
        self.ui.setupUi(self.manual_gui)
        self.manual_gui.show()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindowClass()
    myWindow.show()
    sys.exit(app.exec_())