import socket
import threading
import queue

MAIN_SERVER_IP = "0.0.0.0"
#ADMIN_IP = "192.168.0.24"
#VIDEO_IP = "192.168.0.28"
BUFFER_SIZE = 1024
UDP_BUFFER_SIZE = 65536

# 5000 -> UDP
# 6000 -> TCP
UDP_PORTS = {
    "U_RPI_MAIN": 5000,     
    "U_MAIN_AI": 5001,          
    "U_AI_MAIN": 5002,
    "U_MAIN_ADMIN": 5003       
}
TCP_PORTS = {
    "T_RPI": 6000,
    "T_VIDEO": 6001,
    "T_AUDIO": 6002,
    "T_ADMIN": 6003
}

PORT_NAME = {
    5000: "U_RPI_MAIN",
    5001: "U_MAIN_AI",
    5002: "U_AI_MAIN",
    5003: "U_ADMIN",
    6000: "T_RPI",
    6001: "T_VIDEO",
    6002: "T_AUDIO",
    6003: "T_ADMIN"
}

def receive_data(client_socket, client_id, send_queue):
    while True:
        try:
            data = client_socket.recv(BUFFER_SIZE)
            if not data:
                break
            print(f"[TCP] Client {client_id} sent: {data.decode()}")
            if client_id == "ADMIN":
                processed_data = f"Processed: {data.decode()}".encode()  # ADMIN 명령 가공
                send_queue.put(processed_data)
            else:
                # 로봇 명령 가공 필요 시 추가
                send_queue.put(data)
        except Exception as e:
            print(f"[TCP] Receive Error with Client {client_id}: {e}")
            break

def send_data(client_socket, client_id, send_queue):
    while True:
        try:
            data_to_send = send_queue.get()
            client_socket.sendall(data_to_send)
            print(f"[TCP] Sent to Client {client_id}: {data_to_send.decode()}")
            send_queue.task_done()
        except Exception as e:
            print(f"[TCP] Send Error with Client {client_id}: {e}")
            break

def handle_tcp_client(client_socket, address, client_id, robot_queue = None):
    print(f"[TCP] Client_ID: {client_id}    Address: {address}")
    send_queue = queue.Queue()

    if client_id in ["VIDEO", "AUDIO"]:  # VIDEO, AUDIO -> 로봇
        recv_thread = threading.Thread(
            target = receive_data,
            args = (client_socket, client_id, robot_queue),
            daemon = True
        )
        recv_thread.start()
    elif client_id == "RPI":  # 로봇으로 전송
        send_thread = threading.Thread(
            target = send_data,
            args = (client_socket, client_id, robot_queue),
            daemon = True
        )
        send_thread.start()
    elif client_id == "ADMIN":  # ADMIN <-> 메인
        recv_thread = threading.Thread(
            target = receive_data,
            args = (client_socket, client_id, send_queue),
            daemon = True
        )
        send_thread = threading.Thread(
            target = send_data,
            args = (client_socket, client_id, send_queue),
            daemon = True
        )
        recv_thread.start()
        send_thread.start()

    while threading.active_count() > 1:
        pass

    print(f"[TCP] Client {client_id} disconnected")
    client_socket.close()

def start_tcp_server(port, client_id, robot_queue = None):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((MAIN_SERVER_IP, port))
    server_socket.listen(5)
    print(f"[TCP] Server listening on {MAIN_SERVER_IP}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(
            target = handle_tcp_client,
            args = (client_socket, addr, client_id, robot_queue),
            daemon = True
        ).start()

def udp_server(receive_port, send_port, send_ip):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind((MAIN_SERVER_IP, receive_port))
    print(f"[UDP] Server listening on {MAIN_SERVER_IP}:{receive_port}")

    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_address = (send_ip, send_port)
    print(f"[UDP] Forwarding from {PORT_NAME[receive_port]} --> {PORT_NAME[send_port]} at {send_ip}:{send_port}")

    while True:
        try:
            data, addr = udp_socket.recvfrom(UDP_BUFFER_SIZE)
            print(f"[UDP] Received {len(data)} bytes from {addr}")
            send_socket.sendto(data, send_address)
            print(f"[UDP] Sent {len(data)} bytes to {send_address}")
        except Exception as e:
            print(f"[UDP] Error: {e}")
            break
    udp_socket.close()
    send_socket.close()


def order_test(client_socket, client_id, send_queue):
    while True:
        try:
            command = input(f"[{client_id}] 명령 입력 (종료: 'exit'): ")
            if command.lower() == "exit":
                print(f"[{client_id}] 명령 입력 종료")
                break
            processed_command = command.encode()  # 문자열을 바이트로 변환
            send_queue.put(processed_command)
            print(f"[{client_id}] Queue에 명령 추가: {command}")
        except Exception as e:
            print(f"[{client_id}] 입력 오류: {e}")
            break


if __name__ == "__main__":
    robot_queue = queue.Queue()  # VIDEO/AUDIO -> RPI

    # TCP 서버
    tcp_threads = []
    for client_id, port in TCP_PORTS.items():
        t = threading.Thread(
            target = start_tcp_server,
            args = (port, client_id[2:], robot_queue),
            daemon = True
        )
        tcp_threads.append(t)
        t.start()

    # UDP 사용 x
    # UDP 서버: 로봇 -> 비디오 AI, 비디오 AI -> ADMIN
    """
    udp_threads = []
    udp_threads.append(threading.Thread(
        target=udp_server,
        args=(UDP_PORTS["U_RPI_MAIN"], UDP_PORTS["U_MAIN_AI"], VIDEO_IP),  # 로봇 -> 비디오 AI
        daemon=True
    ))
    udp_threads.append(threading.Thread(
        target=udp_server,
        args=(UDP_PORTS["U_AI_MAIN"], UDP_PORTS["U_MAIN_ADMIN"], ADMIN_IP),  # 비디오 AI -> ADMIN
        daemon=True
    ))
    for t in udp_threads:
        t.start()
    """

    order_thread = threading.Thread(
        target=order_test,
        args=(None, "MANUAL", robot_queue),  # client_socket 없으므로 None, client_id는 "MANUAL"로
        daemon=True
    )
    order_thread.start()


    print("[Main] All servers are running.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n[Main] Shutting down.")

