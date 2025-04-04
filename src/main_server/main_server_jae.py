import socket
import threading

##################################################
# 포트 설정
##################################################
HOST = '0.0.0.0'
MAIN_SERVER_PORT = 5001

connections = {}

##################################################
# TCP소켓 연결받기
##################################################
def handle_client(conn, addr, role):
    print(f"[+] {role} 연결됨: {addr}")
    
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print(f"[-] {role} 연결 종료")
                break

            message = data.decode().strip()
            print(f"[{role}] 수신: {message}")

            # ai서버로 부터 올 때
            if role == 'AI':
                if message in ['STOP', 'LEFT_MOVE', 'RIGHT_MOVE']:
                    if 'RPI' in connections:
                        connections['RPI'].send(data)
                elif message in ['REC_ON', 'REC_OFF']:
                    if 'GUI' in connections:
                        connections['GUI'].send(data)
            # gui로 부터 올 때
            elif role == 'GUI':
                if 'RPI' in connections:
                    connections['RPI'].send(data)

        except Exception as e:
            print(f"[!] {role} 처리 중 예외: {e}")
            break

    conn.close()


##################################################
# MAIN
##################################################
if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((HOST, MAIN_SERVER_PORT))
    server_socket.listen(3)

    print("[*] 서버 시작됨. 연결 대기 중...")

    # 1. AI 서버 연결
    ai_conn, ai_addr = server_socket.accept()
    connections['AI'] = ai_conn
    threading.Thread(target = handle_client, args=(ai_conn, ai_addr, 'AI')).start()

    # 2. GUI 연결
    gui_conn, gui_addr = server_socket.accept()
    connections['GUI'] = gui_conn
    threading.Thread(target = handle_client, args=(gui_conn, gui_addr, 'GUI')).start()

    # 3. 라즈베리파이 연결
    rpi_conn, rpi_addr = server_socket.accept()
    connections['RPI'] = rpi_conn
    threading.Thread(target = handle_client, args=(rpi_conn, rpi_addr, 'RPI')).start()

