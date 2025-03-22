import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QDateEdit
from PyQt5.QtGui import QPixmap
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDate
from io import BytesIO

# UI 파일 로드
ui_file = "/path/to/graph_gui.ui"  # 🔥 경로 변경 필요
log_gui = "/path/to/log_gui.ui"
main_gui = "/path/to/main_gui.ui"

Ui_Dialog, _ = uic.loadUiType(ui_file)

class GraphGUI(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 날짜 설정
        today = QDate.currentDate()
        self.start_date.setDate(today)
        self.end_date.setDate(today)
        self.end_date.setEnabled(False)

        self.start_date.dateChanged.connect(self.toggle_end_date)
        self.start_date.dateChanged.connect(self.update_end_date_minimum)

        # 🔥 버튼 클릭 시 그래프 업데이트
        self.main_btn.clicked.connect(self.back_to_main)
        self.back_btn.clicked.connect(self.back_to_log)

        # 초기 그래프 표시
        self.update_graphs()

    def update_end_date_minimum(self):
        """start_date가 변경되면 end_date의 최소값을 변경"""
        start_date_value = self.start_date.date()
        self.end_date.setMinimumDate(start_date_value)

    def toggle_end_date(self):
        """start_date가 변경되면 end_date 활성화 여부 설정"""
        if self.start_date.date() != QDate(1970, 1, 1):
            self.end_date.setEnabled(True)
        else:
            self.end_date.setEnabled(False)

    def generate_bar_plot(self):
        """Matplotlib을 사용하여 막대 그래프 생성"""
        x = np.arange(5)
        y = np.random.randint(1, 10, 5)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(x, y, color='skyblue')

        ax.set_title("Bar Plot")
        ax.set_xlabel("Categories")
        ax.set_ylabel("Values")

        pixmap = self.convert_plot_to_pixmap(fig)
        plt.close(fig)  # 🔥 메모리 누수 방지
        return pixmap

    def generate_line_plot(self):
        """Matplotlib을 사용하여 선 그래프 생성"""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, y, color='red')

        ax.set_title("Line Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        pixmap = self.convert_plot_to_pixmap(fig)
        plt.close(fig)  # 🔥 메모리 누수 방지
        return pixmap

    def convert_plot_to_pixmap(self, fig):
        """Matplotlib Figure를 QPixmap으로 변환"""
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())

        return pixmap

    def update_graphs(self):
        """QLabel에 그래프 표시"""
        bar_pixmap = self.generate_bar_plot()
        line_pixmap = self.generate_line_plot()

        # 🔥 `bar_label` → Bar Plot 표시
        self.bar_label.setPixmap(bar_pixmap.scaled(self.bar_label.size(), aspectRatioMode=1))
        
        # 🔥 `line_label` → Line Plot 표시
        self.line_label.setPixmap(line_pixmap.scaled(self.line_label.size(), aspectRatioMode=1))
    
    # Change the method like this:
    def back_to_main(self):
        from main_gui import MainGUI  # 👈 import here
        self.main_window = MainGUI()
        self.main_window.show()
        self.close()

    def back_to_log(self):
        from log_gui import LogGUI
        self.log_window = LogGUI()
        self.log_window.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GraphGUI()
    gui.show()
    sys.exit(app.exec_())
