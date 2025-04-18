import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import mysql.connector
from main_gui import MainGUI
from dotenv import load_dotenv
import os

load_dotenv()

loginGUI = os.environ.get("PATH_TO_LOGIN_GUI")
mainGUI = os.environ.get("PATH_TO_MAIN_GUI")

HOST = os.environ.get("MYSQL_HOST")
USER = os.environ.get("MYSQL_USER")
PASSWD = os.environ.get("MYSQL_PASSWD")
DB_NAME = os.environ.get("DB_NAME")

logInUi = uic.loadUiType(loginGUI)[0]
mainUi = uic.loadUiType(mainGUI)[0]




class LogInWindow(QMainWindow, logInUi):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.remote = mysql.connector.connect(
            host=HOST,
            user=USER,
            password=PASSWD,
            database=DB_NAME
        )

        self.lineEdit1.setEchoMode(QLineEdit.Normal)
        self.lineEdit2.setEchoMode(QLineEdit.Password)

        self.Login_btn.clicked.connect(self.open_main_window)
        self.lineEdit2.returnPressed.connect(self.open_main_window)



    def open_main_window(self):
        """ MySQL 로그인 검증 후 Main.py 실행 """
        id = self.lineEdit1.text()
        passwd = self.lineEdit2.text()

        cursor = self.remote.cursor()
        cursor.execute(f"SELECT * FROM Users WHERE id = %s and passwd = SHA2(%s, 256)", (id, passwd))
        result = cursor.fetchone()

        if result:
            self.main_window = MainGUI()  # Main.py의 WindowClass 실행
            self.main_window.show()  # Main.ui를 실행
            self.close()  # 현재 로그인 창 닫기
        else:
            QMessageBox.warning(self, '로그인 실패', '아이디 또는 비밀번호가 틀렸습니다. 확인 후 다시 시도해주세요.')



if __name__=="__main__":
    app = QApplication(sys.argv)
    loginWindow = LogInWindow()
    loginWindow.show()
    sys.exit(app.exec_())