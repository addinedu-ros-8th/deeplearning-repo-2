import sys
import cv2
import socket
import threading
import numpy as np
import queue
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QHeaderView
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QDate, QTimer, Qt
from PyQt5 import uic
from manual_control_gui import ManualWindowClass




# Load the UI file
mainUi = uic.loadUiType("/home/lim/dev_ws/mldl_project/src/hosbot.ui")[0]
manualUi = uic.loadUiType("/home/lim/dev_ws/deeplearning-repo-2/src/admin_pc/manual_control_gui.ui")[0]







# UDP Video Streaming Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 6001
BUFFER_SIZE = 65536  # Max UDP packet size






class UDPWebcamFrame(QFrame):
    """Custom QFrame to receive and display the UDP video stream."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Queue for thread-safe frame handling
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Start UDP thread to receive video frames
        self.udp_thread = threading.Thread(target=self.receive_frames, daemon=True)
        self.udp_thread.start()

        # Timer to refresh displayed frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update UI every 30ms

        self.latest_frame = None  # Store the latest received frame



    def receive_frames(self):
        """Receives video frames over UDP in a separate thread."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        packet_buffer = {}  # Buffer for reassembling frames

        while True:
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
                if data == b"END_OF_FILE":
                    print("[UDP] Video stream ended.")
                    break

                # Extract metadata: "frame_id,packet_num,total_packets||data"
                try:
                    header, packet_data = data.split(b"||", 1)
                    frame_id, packet_num, total_packets = map(int, header.decode().split(","))
                except ValueError:
                    print("[UDP] Corrupt packet received, skipping...")
                    continue

                # Store packet in buffer
                if frame_id not in packet_buffer:
                    packet_buffer[frame_id] = [None] * total_packets
                packet_buffer[frame_id][packet_num] = packet_data

                # If full frame is received, decode and queue it
                if None not in packet_buffer[frame_id]:
                    full_frame_data = b"".join(packet_buffer[frame_id])
                    frame = cv2.imdecode(np.frombuffer(full_frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    del packet_buffer[frame_id]  # Free the buffer for this frame

            except socket.timeout:
                pass  # Continue if no data received



    def update_frame(self):
        """Updates the displayed frame from the frame queue."""
        if not self.frame_queue.empty():
            self.latest_frame = self.frame_queue.get()
        self.update()  # Trigger repaint



    def paintEvent(self, event):
        """Override paintEvent to draw the latest received frame."""
        if self.latest_frame is not None:
            painter = QPainter(self)
            h, w, ch = self.latest_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.latest_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()



    def closeEvent(self, event):
        """Close UDP thread and timer properly."""
        # Optionally, you can add cleanup for the UDP thread if needed.
        self.timer.stop()
        event.accept()






class MainWindowClass(QMainWindow, mainUi):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
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

        self.setWindowTitle("Hosbot")

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
        self.manual_control_window = ManualWindowClass()  # manual_control_gui.py의 WindowClass 실행
        self.manual_control_window.show()  # manual_control_gui.ui를 실행
        self.close()  # 현재 main 창 닫기






if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindowClass()
    myWindow.show()
    sys.exit(app.exec_())