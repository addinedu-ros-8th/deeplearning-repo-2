import sys
import socket
import threading
import numpy as np
import queue
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import uic
import cv2

UDP_IP = 'local IP'
UDP_PORT = "your port Num"
BUFFER_SIZE = 65536

class SharedUDPReceiver:
    _instance = None

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.listeners = []
        self.running = True
        self.thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.thread.start()
        print(f"[SharedUDPReceiver] Bound to {UDP_IP}:{UDP_PORT}")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = SharedUDPReceiver()
        return cls._instance

    def register_listener(self, listener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    def receive_loop(self):
        packet_buffer = {}
        while self.running:
            try:
                data, addr = self.sock.recvfrom(BUFFER_SIZE)
                # Process and reassemble frame (same as before) ...
                # When a full frame is ready:
                frame = ...  # decoded frame
                for listener in self.listeners:
                    listener(frame)
            except Exception as e:
                print("Error:", e)


class WebcamFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_queue = queue.Queue(maxsize=5)
        self.latest_frame = None

        # Instead of starting its own UDP thread, register with the shared receiver:
        SharedUDPReceiver.instance().register_listener(self.on_frame_received)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def on_frame_received(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def update_frame(self):
        if not self.frame_queue.empty():
            self.latest_frame = self.frame_queue.get()
        self.update()

    def paintEvent(self, event):
        if self.latest_frame is not None:
            painter = QPainter(self)
            h, w, ch = self.latest_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.latest_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()
