import socket

# Admin Client에서 수신할 포트
MAIN_SERVER_PORT = 6002
# Raspberry Pi로 보낼 포트
RASPBERRY_PI_IP = "172.29.146.150"  # Raspberry Pi IP 주소
RASPBERRY_PI_PORT = 6001

def forward_to_raspberry(message):
    """Raspberry Pi로 메시지 전달"""
    try:
        pi_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        pi_sock.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
        pi_sock.sendall(message.encode("utf-8"))
        pi_sock.close()
        print(f"[→] Sent to Raspberry Pi: {message}")
    except Exception as e:
        print(f"[!] Failed to send to Raspberry Pi: {e}")

def handle_admin_client():
    """Admin Client에서 명령을 받아 Raspberry Pi로 전달"""
    main_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    main_sock.bind(("0.0.0.0", MAIN_SERVER_PORT))
    main_sock.listen(5)

    print(f"[*] Listening for commands on port {MAIN_SERVER_PORT}...")

    while True:
        client, addr = main_sock.accept()
        print(f"[*] Connection from {addr}")

        try:
            while True:
                data = client.recv(1024).decode("utf-8")
                if not data:
                    break
                
                print(f"[Admin Client] Received: {data}")

                # Raspberry Pi로 전달
                forward_to_raspberry(data)
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            client.close()

if __name__ == "__main__":
    handle_admin_client()
