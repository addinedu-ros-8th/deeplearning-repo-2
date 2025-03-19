import socket
from gpiozero import Motor, PWMOutputDevice
from time import sleep

# Raspberry Pi에서 수신할 포트
RASPBERRY_PI_PORT = 6001

# 모터 제어 핀 설정
motor1a = 23  # 모터 1 방향 제어
motor1b = 22  # 모터 1 방향 제어
motor2a = 27  # 모터 2 방향 제어
motor2b = 17  # 모터 2 방향 제어
motor1_pwm = PWMOutputDevice(24)  # 모터 1 속도 제어 (PWM)
motor2_pwm = PWMOutputDevice(18)  # 모터 2 속도 제어 (PWM)

motor3a = 16   # 모터 3 방향 제어
motor3b = 12   # 모터 3 방향 제어
motor4a = 6    # 모터 4 방향 제어
motor4b = 5    # 모터 4 방향 제어
motor3_pwm = PWMOutputDevice(26)  # 모터 3 속도 제어 (PWM)
motor4_pwm = PWMOutputDevice(11)  # 모터 4 속도 제어 (PWM)

# Motor 객체 생성 (각각의 모터에 대한 방향 제어)
motor1 = Motor(motor1a, motor1b)
motor2 = Motor(motor2a, motor2b)
motor3 = Motor(motor3a, motor3b)
motor4 = Motor(motor4a, motor4b)

# 모터 속도 설정 (0.0 - 1.0)
motor1_pwm.value = 1.0  # 속도 100%
motor2_pwm.value = 1.0  # 속도 100%
motor3_pwm.value = 1.0  # 속도 100%
motor4_pwm.value = 1.0  # 속도 100%

def motor_control(command):
    """명령에 맞춰 모터를 제어하는 함수"""
    if command == "FORWARD":  # 전진
        motor1.backward()
        motor2.backward()
        motor3.backward()
        motor4.backward()
        print("모터들이 전진합니다.")
    
    elif command == "BACKWARD":  # 후진
        motor1.forward()
        motor2.forward()
        motor3.forward()
        motor4.forward()
        print("모터들이 후진합니다.")
    
    elif command == "STOP":  # 정지
        motor1.stop()
        motor2.stop()
        motor3.stop()
        motor4.stop()
        print("모터들이 정지했습니다.")

    elif command == "RIGHT_TURN":  # 왼쪽 회전
        motor1.backward()
        motor2.forward()
        motor3.backward()
        motor4.forward()
        print("오른쪽으로 회전중.")
    
    elif command == "LEFT_TURN":  # 오른쪽 회전
        motor1.forward()
        motor2.backward()
        motor3.forward()
        motor4.backward()
        print("왼쪽으로 회전중.")

    elif command == "LEFT_MOVE":  # 왼쪽 이동
        motor1.backward()
        motor2.backward()
        motor3.forward()
        motor4.forward()
        print("왼쪽으로 이동 중.")
    
    elif command == "RIGHT_MOVE":  # 오른쪽 이동
        motor1.forward()
        motor2.forward()
        motor3.backward()
        motor4.backward()
        print("오른쪽으로 이동 중.")
    
    else:
        print(f"[?] Unknown command: {command}")

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
                motor_control(data)
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            client.close()

if __name__ == "__main__":
    handle_main_server()
