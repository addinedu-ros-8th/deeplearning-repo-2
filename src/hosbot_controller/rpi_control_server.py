import socket
import cv2
import threading
from gpiozero import Motor, PWMOutputDevice

# ==============================
#  Configuration Settings
# ==============================
SERVER_IP = "192.168.65.82"  # 서버 IP
UDP_PORT = 5000
PACKET_SIZE = 60000  # UDP 패킷 크기

RASPBERRY_PI_PORT = 5001

# ==============================
#  UDP Webcam Streaming 설정
# ==============================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# ==============================
#  TCP Motor Control 설정
# ==============================
# 모터 핀 설정
motor1 = Motor(23, 22)
motor2 = Motor(27, 17)
motor3 = Motor(16, 12)
motor4 = Motor(6, 5)

motor1_pwm = PWMOutputDevice(24)  # 모터 1 속도 제어
motor2_pwm = PWMOutputDevice(18)  # 모터 2 속도 제어
motor3_pwm = PWMOutputDevice(26)  # 모터 3 속도 제어
motor4_pwm = PWMOutputDevice(11)  # 모터 4 속도 제어

# 모터 속도 기본값 설정
motor1_pwm.value = 1.0
motor2_pwm.value = 1.0
motor3_pwm.value = 1.0
motor4_pwm.value = 1.0

# ==============================
#  모터 제어 함수
# ==============================
def motor_control(command):
    if command == "FORWARD":  # 전진
        motor1.backward()
        motor2.backward()
        motor3.backward()
        motor4.backward()
        print("전진")

    elif command == "BACKWARD":  # 후진
        motor1.forward()
        motor2.forward()
        motor3.forward()
        motor4.forward()
        print("후진")

    elif command == "STOP":  # 정지
        motor1.stop()
        motor2.stop()
        motor3.stop()
        motor4.stop()
        print("정지")

    elif command == "RIGHT_TURN":  # 오른쪽 회전
        motor1.backward()
        motor2.forward()
        motor3.backward()
        motor4.forward()
        print("오른쪽 회전")

    elif command == "LEFT_TURN":  # 왼쪽 회전
        motor1.forward()
        motor2.backward()
        motor3.forward()
        motor4.backward()
        print("왼쪽 회전")

    elif command == "LEFT_MOVE":  # 왼쪽 이동
        motor1.backward()
        motor2.backward()
        motor3.forward()
        motor4.forward()
        print("왼쪽 이동")

    elif command == "RIGHT_MOVE":  # 오른쪽 이동
        motor1.forward()
        motor2.forward()
        motor3.backward()
        motor4.backward()
        print("오른쪽 이동")

    else:
        print(f"[?] Unknown command: {command}")

# ==============================
#  UDP 웹캠 스트리밍 스레드
# ==============================
def stream_webcam():
    global cap
    print(f"[UDP] Streaming webcam to {SERVER_IP}:{UDP_PORT}")

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 JPEG로 인코딩
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        encoded_bytes = encoded_frame.tobytes()

        # UDP 패킷으로 나눠서 전송
        total_packets = len(encoded_bytes) // PACKET_SIZE + 1
        for i in range(total_packets):
            start = i * PACKET_SIZE
            end = start + PACKET_SIZE
            packet_data = encoded_bytes[start:end]
            
            header = f"{frame_id},{i},{total_packets}".encode()
            sock.sendto(header + b"||" + packet_data, (SERVER_IP, UDP_PORT))

        frame_id += 1
        cv2.waitKey(1)

    # END 신호 전송
    sock.sendto(b"END_OF_FILE", (SERVER_IP, UDP_PORT))
    print("[UDP] Webcam stream ended.")
    cap.release()
    sock.close()

# ==============================
#  TCP 모터 제어 스레드
# ==============================
def handle_main_server():
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
                motor_control(data)
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            client.close()

# ==============================
#  멀티스레드 실행
# ==============================
if __name__ == "__main__":
    thread1 = threading.Thread(target=stream_webcam, daemon=True)
    thread2 = threading.Thread(target=handle_main_server, daemon=True)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
