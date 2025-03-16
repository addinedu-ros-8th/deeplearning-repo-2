import socket
import threading

# Admin GUI에서 명령을 받을 포트
ADMIN_PORT = 7001
# 메인 서버로 보낼 포트
MAIN_SERVER_IP = "172.29.146.239"  # 메인 서버 IP 주소
MAIN_SERVER_PORT = 6002

def handle_admin_commands():
    """Admin GUI에서 명령을 받아 메인 서버로 전달"""
    admin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    admin_sock.bind(("0.0.0.0", ADMIN_PORT))
    admin_sock.listen(5)

    print(f"[*] Listening for admin commands on port {ADMIN_PORT}...")

    while True:
        client, addr = admin_sock.accept()
        print(f"[*] Connection from {addr}")

        try:
            while True:
                data = client.recv(1024).decode("utf-8")
                if not data:
                    break
                
                print(f"[Admin GUI] Received: {data}")

                # 메인 서버로 전달
                send_to_main_server(data)
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            client.close()

def send_to_main_server(message):
    """메인 서버로 명령 전송"""
    try:
        main_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        main_sock.connect((MAIN_SERVER_IP, MAIN_SERVER_PORT))
        main_sock.sendall(message.encode("utf-8"))
        main_sock.close()
        print(f"[→] Sent to Main Server: {message}")
    except Exception as e:
        print(f"[!] Failed to send to main server: {e}")

if __name__ == "__main__":
    threading.Thread(target=handle_admin_commands, daemon=True).start()
    
    while True:
        command = input("Enter command to send to main server: ")
        send_to_main_server(command)
