import os
import socket
import threading
from dotenv import load_dotenv
import mysql.connector
import datetime

load_dotenv()

##################################################
# 포트 설정
##################################################
HOST = os.environ.get("MAIN_IP")
MAIN_SERVER_PORT = os.environ.get("SUB_PORT")

connections = {}

DB_HOST = os.environ.get("MYSQL_HOST")
DB_USER = os.environ.get("MYSQL_USER")
DB_PASSWD = os.environ.get("MYSQL_PASSWD")
DB_NAME = os.environ.get("DB_NAME")


##################################################
# DB연결
##################################################
def connectLocal():
    local = mysql.connector.connect(
        host = DB_HOST,
        user = DB_USER,
        password = DB_PASSWD,
        database = DB_NAME
    )
    return local


##################################################
# TCP소켓 연결받기
##################################################
def handle_client(conn, addr, role):
    print(f"[+] {role} 연결됨: {addr}")
    
    local = connectLocal()
    cursor = local.cursor()

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print(f"[-] {role} 연결 종료")
                break

            message = data.decode().strip()
            print(f"[{role}] 수신: {message}")
            
            """
            if role == 'AI':
                if message in ['STOP', 'LEFT_MOVE', 'RIGHT_MOVE']:
                    if 'RPI' in connections:
                        connections['RPI'].send(data)
                elif message in ['REC_ON:', 'REC_OFF:']:
                    if 'GUI' in connections:
                        connections['GUI'].send(data)
            """

            if role == 'AI':
                if 'GUI' in connections:
                    connections['GUI'].send(data)

            
            elif role == 'GUI':
                if message in ['STOP', 'LEFT_MOVE', 'RIGHT_MOVE', 'FORWARD', 'BACKWARD', 'LEFT_TURN', 'RIGHT_TURN']:
                    print("from gui", message)
                    if 'RPI' in connections:
                        connections['RPI'].send(data)                
                else:
                    print(f"[DB] 파일명으로 저장: {message}")
                    insert_query = "INSERT INTO file_log (filename) VALUES (%s)"
                    cursor.execute(insert_query, (message,))
                    local.commit()

                    ret = message.split(':')

                    # eventId 매핑
                    event_map = {
                        'fire': 1,
                        'fighting': 2,
                        'lying': 3,
                        'smoking': 4
                    }

                    event_type = ret[1].lower()
                    event_id = event_map.get(event_type, None)

                    if event_id is not None:
                        insert_event_query = """
                        INSERT INTO Event_Log (placeId, robotId, eventId, videoPath, createDate)
                        VALUES (%s, %s, %s, %s, %s)
                        """

                        cursor.execute(
                            insert_event_query,
                            (
                                None,  # placeId = NULL
                                1,     # robotId
                                event_id,     # eventId
                                ret[0],       # videoPath
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # createDate
                            )
                        )
                        local.commit()
                        print("[DB] Event_Log에 insert 완료")
                    else:
                        print(f"[DB] 알 수 없는 event type: {event_type}")


        except Exception as e:
            print(f"[!] {role} 처리 중 예외: {e}")
            break

    conn.close()
    cursor.close()
    local.close()


##################################################
# MAIN
##################################################
if __name__ == "__main__":
    local = connectLocal()  # 테스트 연결만, 이후 사용 X
    local.close()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, MAIN_SERVER_PORT))
    server_socket.listen(3)

    print("[*] 서버 시작됨. 연결 대기 중...")

    # 1. GUI 연결
    gui_conn, gui_addr = server_socket.accept()
    connections['GUI'] = gui_conn
    threading.Thread(target=handle_client, args=(gui_conn, gui_addr, 'GUI')).start()

    # 2. AI 서버 연결
    ai_conn, ai_addr = server_socket.accept()
    connections['AI'] = ai_conn
    threading.Thread(target=handle_client, args=(ai_conn, ai_addr, 'AI')).start()

    """
    # 3. 라즈베리파이 연결
    rpi_conn, rpi_addr = server_socket.accept()
    connections['RPI'] = rpi_conn
    threading.Thread(target=handle_client, args=(rpi_conn, rpi_addr, 'RPI')).start()
    """
