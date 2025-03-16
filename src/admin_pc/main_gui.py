import sys
import cv2
import socket
import threading
import numpy as np
import queue
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QHeaderView
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QDate, QTimer, Qt
from PyQt5 import uic
from udp_receiver import UDPWebcamFrame




# Load the UI file
mainUi = uic.loadUiType("/home/lim/dev_ws/deeplearning-repo-2/src/admin_pc/main_gui.ui")[0]





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
        self.webcam_frame = UDPWebcamFrame(self)
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



    def closeEvent(self, event):
        """Ensure proper cleanup on window close."""
        # Attempt to join the UDP thread with a timeout (if it's still running)
        self.webcam_frame.udp_thread.join(1)
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