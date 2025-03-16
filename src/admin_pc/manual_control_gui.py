import sys
import socket
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5 import uic
from PyQt5 import QtWidgets
from udp_receiver import WebcamFrame




ADMIN_CLIENT_IP = 'local IP'  # Admin Client (7001)
ADMIN_CLIENT_PORT = "your port Num"




ui_file_path = "/path/to/manual_control_gui.ui"
form_class, base_class = uic.loadUiType(ui_file_path)




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




class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.main_btn.setVisible(True)

        self.pushButton.clicked.connect(lambda: self.send_command("FORWARD"))
        self.pushButton_3.clicked.connect(lambda: self.send_command("LEFT"))
        self.pushButton_4.clicked.connect(lambda: self.send_command("RIGHT"))
        self.pushButton_5.clicked.connect(lambda: self.send_command("BACKWARD"))
        self.pushButton_6.clicked.connect(lambda: self.send_command("STOP"))
        self.pushButton_7.clicked.connect(lambda: self.send_command("LEFT_TURN"))
        self.pushButton_8.clicked.connect(lambda: self.send_command("RIGHT_TURN"))

        print("main_btn isEnabled:", self.main_btn.isEnabled(), "isVisible:", self.main_btn.isVisible())
        self.main_btn.clicked.connect(lambda: print("main_btn clicked"))
        self.main_btn.clicked.connect(self.open_main_gui)

        # Replace the existing QFrame (from UI) with our UDPWebcamFrame
        self.webcam_frame = WebcamFrame(self)
        self.webcam_frame.setGeometry(self.frame.geometry())  # Match the size and position of the UI frame
        self.webcam_frame.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.frame.hide()  # Hide the original placeholder frame

    def update_video_frame(self, qimg):
        self.video_frame.set_image(qimg)

    def send_command(self, command):
        print(f"Sending command: {command}")
        self.admin_client_thread = AdminClientThread(command)
        self.admin_client_thread.start()

    def closeEvent(self, event):
        # """ Close UDP thread properly when exiting """
        # self.webcam_frame.udp_thread.join(1)  # Stop UDP thread safely
        event.accept()

    def open_main_gui(self):
        print("opening main gui")
        from main_gui import MainWindowClass
        self.main_gui = QtWidgets.QMainWindow()
        self.ui = MainWindowClass()
        self.ui.setupUi(self.main_gui)
        self.main_gui.show()
        self.close()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())