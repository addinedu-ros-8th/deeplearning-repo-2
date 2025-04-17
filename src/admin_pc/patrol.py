import socket
import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys

# YOLO 모델 로드
model = YOLO("/home/shim/dev_ws/runs/detect/train/weights/best.pt")

# UDP 소켓 설정 (비디오 수신용)
udp_ip = "0.0.0.0"
udp_port = 5000
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
udp_sock.bind((udp_ip, udp_port))
buffer_size = 65507

# TCP 설정 (결과 전송용)
tcp_ip = "172.24.125.150"
tcp_port = 6001
tcp_sock = None

# 이전 상태 저장 변수
previous_objects = set()
no_detection_start_time = None  # 감지 안 되는 상태 시작 시간
last_left_turn_time = 0  # 마지막 LEFT_TURN 감지 시간
LEFT_TURN_COOLDOWN = 2  # LEFT_TURN 쿨타임 (초)

def connect_tcp():
    """TCP 서버에 연결하는 함수"""
    global tcp_sock
    while True:
        try:
            tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_sock.connect((tcp_ip, tcp_port))
            print(f"[TCP] 서버 ({tcp_ip}:{tcp_port})에 연결됨")
            return
        except Exception as e:
            print(f"[TCP] 연결 실패: {e}. 3초 후 재시도...")
            time.sleep(3)

# TCP 서버 연결 시도
connect_tcp()

print("[UDP] 수신 대기 중...")

first_packet_received_time = None
forward_sent = False

while True:
    try:
        # 데이터 수신 (UDP)
        packet, _ = udp_sock.recvfrom(buffer_size)

        # 첫 패킷 수신 시간 기록
        if first_packet_received_time is None:
            first_packet_received_time = time.time()
            print("[UDP] 첫 비디오 패킷 수신")

        # 1초 후 FORWARD 전송
        if not forward_sent and time.time() - first_packet_received_time >= 1.0:
            try:
                tcp_sock.sendall("FORWARD".encode())
                print(f"[TCP] 초기 신호 전송 (1초 후): FORWARD")
                previous_objects = {"FORWARD"}
                forward_sent = True
            except Exception as e:
                print(f"[TCP] 초기 전송 오류: {e}")
                tcp_sock.close()
                connect_tcp()

        frame = cv2.imdecode(np.frombuffer(packet, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print("[UDP] 프레임 복원 실패")
            continue

        # YOLO 감지 실행
        results = model(frame)
        detected_objects = set()
        current_time = time.time()

        for result in results:
            keep_indices = result.boxes.conf >= 0.9
            result.boxes = result.boxes[keep_indices]

            for i in result.boxes.cls.tolist():
                obj_name = result.names[i]
                if obj_name == "Stop":
                    obj_name = "STOP"
                    print("[YOLO] STOP 감지됨. 프로그램 종료.")
                    tcp_sock.sendall("STOP".encode())  # 필요 시 STOP 전송
                    tcp_sock.close()
                    udp_sock.close()
                    sys.exit()  # 프로그램 종료

                elif obj_name == "Left hand curve":
                    obj_name = "LEFT_TURN"
                    # 쿨타임 검사
                    if current_time - last_left_turn_time < LEFT_TURN_COOLDOWN:
                        continue  # 무시
                    last_left_turn_time = current_time

                detected_objects.add(obj_name)

        # 감지된 객체가 없을 경우 1.5초 동안 유지 확인 후 FORWARD 설정
        if not detected_objects:
            if no_detection_start_time is None:
                no_detection_start_time = current_time
            elif current_time - no_detection_start_time >= 2:
                detected_objects.add("FORWARD")
        else:
            no_detection_start_time = None

        # 상태 변경 여부 확인
        if detected_objects != previous_objects:
            print(f"[YOLO] 상태 변경 감지: {detected_objects}")
            try:
                data_to_send = ", ".join(detected_objects)
                tcp_sock.sendall(data_to_send.encode())
                print(f"[TCP] 데이터 전송: {data_to_send}")
                previous_objects = detected_objects
            except Exception as e:
                print(f"[TCP] 전송 오류: {e}")
                tcp_sock.close()
                connect_tcp()

        # 감지된 영상 표시
        frame = results[0].plot()
        cv2.imshow("YOLO Detection", frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print(f"[UDP] 수신 오류: {e}")

# 리소스 정리
udp_sock.close()
tcp_sock.close()
cv2.destroyAllWindows()
