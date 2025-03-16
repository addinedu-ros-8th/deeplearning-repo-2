import socket

# Raspberry Pi에서 수신할 포트
RASPBERRY_PI_PORT = 6001

def handle_main_server():
    """메인 서버에서 명령을 수신하여 처리"""
    pi_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    pi_sock.bind(("0.0.0.0", RASPBERRY_PI_PORT))
    pi_sock.listen(5)

    print(f"[*] Listening for commands on port {RASPBERRY_PI_PORT}...")

    while True:
        client, addr = pi_sock.accept()
        print(f"[*] Connection from {addr}")

        try:
            while True:
                data = client.recv(1024).decode("utf-8")
                if not data:
                    break
                
                print(f"[Main Server] Received: {data}")

                # 받은 명령 처리
                process_command(data)
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            client.close()

def process_command(command):
    """수신된 명령을 처리"""
    if command == "FORWARD":
        print("[→] Moving Forward")
    elif command == "LEFT":
        print("[←] Moving Left")
    elif command == "RIGHT":
        print("[→] Moving Right")
    elif command == "BACKWARD":
        print("[↓] Moving Backward")
    elif command == "STOP":
        print("[X] Stopping")
    elif command == "LEFT_TURN":
        print("[↺] Turning Left")
    elif command == "RIGHT_TURN":
        print("[↻] Turning Right")
    else:
        print(f"[?] Unknown command: {command}")

if __name__ == "__main__":
    handle_main_server()
