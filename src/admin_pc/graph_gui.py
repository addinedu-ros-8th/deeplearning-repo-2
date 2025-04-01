import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QDateEdit
from PyQt5.QtGui import QPixmap
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDate
from io import BytesIO
import mysql.connector
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

GRAPH_GUI = os.environ.get("PATH_TO_GRAPH_GUI")
LOG_GUI = os.environ.get("PATH_TO_LOG_GUI")
MAIN_GUI = os.environ.get("PATH_TO_MAIN_GUI")

HOST = os.environ.get("MYSQL_HOST")
USER = os.environ.get("MYSQL_USER")
PASSWD = os.environ.get("MYSQL_PASSWD")
DB_NAME = os.environ.get("DB_NAME")

# Load UI file
ui_file = GRAPH_GUI
log_gui = LOG_GUI
main_gui = MAIN_GUI

Ui_Dialog, _ = uic.loadUiType(ui_file)

class GraphGUI(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.remote = mysql.connector.connect(
            host=HOST,
            user=USER,
            password=PASSWD,
            database=DB_NAME
        )

        # Set initial date values
        today = QDate.currentDate()
        self.start_date.setDate(today)
        self.end_date.setDate(today)
        self.end_date.setEnabled(False)

        self.start_date.dateChanged.connect(self.toggle_end_date)
        self.start_date.dateChanged.connect(self.update_end_date_minimum)
        self.start_date.dateChanged.connect(self.update_graphs)
        self.end_date.dateChanged.connect(self.update_graphs)

        # Update graph when mode (comboBox) changes
        self.comboBox.currentTextChanged.connect(self.update_graphs)

        self.main_btn.clicked.connect(self.back_to_main)
        self.back_btn.clicked.connect(self.back_to_log)

    def update_end_date_minimum(self):
        """Set minimum date for end_date based on start_date"""
        start_date_value = self.start_date.date()
        self.end_date.setMinimumDate(start_date_value)

    def toggle_end_date(self):
        """Enable end_date only if start_date is set"""
        if self.start_date.date() != QDate(1970, 1, 1):
            self.end_date.setEnabled(True)
        else:
            self.end_date.setEnabled(False)

    def generate_bar_plot(self, mode, start_date_qdate, end_date_qdate):
        event_id_map = {
            "Fight": 1,
            "Fire": 2,
            "Lying": 3
        }

        start_str = start_date_qdate.toString("yyyy-MM-dd")
        end_str = end_date_qdate.toString("yyyy-MM-dd")
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        total_days = (end_date - start_date).days + 1
        date_labels = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(total_days)]

        fig, ax = plt.subplots(figsize=(7, 4))

        if mode == "Select":
            cursor = self.remote.cursor()
            width = 0.2
            x = np.arange(len(date_labels))

            for idx, (event_name, event_id) in enumerate(event_id_map.items()):
                query = """
                    SELECT DATE(createDate) as date, COUNT(*) as count
                    FROM Event_Log
                    WHERE eventId = %s AND DATE(createDate) BETWEEN %s AND %s
                    GROUP BY DATE(createDate)
                """
                cursor.execute(query, (event_id, start_str, end_str))
                result = cursor.fetchall()
                count_dict = {row[0].strftime("%Y-%m-%d"): row[1] for row in result}
                counts = [count_dict.get(day, 0) for day in date_labels]

                ax.bar(x + width * idx, counts, width=width, label=event_name)

            ax.set_xticks(x + width)
            ax.set_xticklabels(date_labels, rotation=45)

            ax.set_title("All Events - Daily Count")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.legend()

        else:
            event_id = event_id_map.get(mode)
            if event_id is None:
                return QPixmap()

            cursor = self.remote.cursor()
            query = """
                SELECT DATE(createDate) as date, COUNT(*) as count
                FROM Event_Log
                WHERE eventId = %s AND DATE(createDate) BETWEEN %s AND %s
                GROUP BY DATE(createDate)
            """
            cursor.execute(query, (event_id, start_str, end_str))
            result = cursor.fetchall()
            count_dict = {row[0].strftime("%Y-%m-%d"): row[1] for row in result}
            counts = [count_dict.get(day, 0) for day in date_labels]

            ax.bar(date_labels, counts, color='skyblue')
            ax.set_title(f"{mode} - Daily Count")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)

        pixmap = self.convert_plot_to_pixmap(fig)
        plt.close(fig)
        return pixmap

    def generate_line_plot(self, mode="default", start_date_qdate=None, end_date_qdate=None):
        event_id_map = {
            "Fight": 1,
            "Fire": 2,
            "Lying": 3
        }

        if start_date_qdate is None or end_date_qdate is None:
            return QPixmap()

        start_str = start_date_qdate.toString("yyyy-MM-dd")
        end_str = end_date_qdate.toString("yyyy-MM-dd")
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        total_days = (end_date - start_date).days + 1
        date_labels = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(total_days)]

        fig, ax = plt.subplots(figsize=(7, 4))
        cursor = self.remote.cursor()

        if mode == "Select":
            for event_name, event_id in event_id_map.items():
                query = """
                    SELECT DATE(createDate) as date, COUNT(*) as count
                    FROM Event_Log
                    WHERE eventId = %s AND DATE(createDate) BETWEEN %s AND %s
                    GROUP BY DATE(createDate)
                """
                cursor.execute(query, (event_id, start_str, end_str))
                result = cursor.fetchall()
                count_dict = {row[0].strftime("%Y-%m-%d"): row[1] for row in result}
                counts = [count_dict.get(day, 0) for day in date_labels]

                ax.plot(date_labels, counts, marker='o', linestyle='-', label=event_name)

            ax.set_title("All Events - Line Graph")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

        else:
            event_id = event_id_map.get(mode)
            if event_id is None:
                return QPixmap()

            query = """
                SELECT DATE(createDate) as date, COUNT(*) as count
                FROM Event_Log
                WHERE eventId = %s AND DATE(createDate) BETWEEN %s AND %s
                GROUP BY DATE(createDate)
            """
            cursor.execute(query, (event_id, start_str, end_str))
            result = cursor.fetchall()
            count_dict = {row[0].strftime("%Y-%m-%d"): row[1] for row in result}
            counts = [count_dict.get(day, 0) for day in date_labels]

            ax.plot(date_labels, counts, color='red', marker='o', linestyle='-')
            ax.set_title(f"{mode} - Trend")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)

        pixmap = self.convert_plot_to_pixmap(fig)
        plt.close(fig)
        return pixmap

    def convert_plot_to_pixmap(self, fig):
        """Convert Matplotlib Figure to QPixmap"""
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())

        return pixmap

    def update_graphs(self):
        selected_option = self.comboBox.currentText()
        start = self.start_date.date()
        end = self.end_date.date()

        bar_pixmap = self.generate_bar_plot(selected_option, start, end)
        line_pixmap = self.generate_line_plot(mode=selected_option, start_date_qdate=start, end_date_qdate=end)

        self.bar_label.setPixmap(bar_pixmap.scaled(self.bar_label.size(), aspectRatioMode=1))
        self.line_label.setPixmap(line_pixmap.scaled(self.line_label.size(), aspectRatioMode=1))

    def back_to_main(self):
        from main_gui import MainGUI
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
