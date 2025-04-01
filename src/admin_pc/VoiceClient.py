import speech_recognition as sr
import socket
import pyttsx3

RASPBERRY_PI_IP = '192.168.28.150'  # Raspberry Pi IP로 변경
RASPBERRY_PI_PORT = 6001

recognizer = sr.Recognizer()
mic = sr.Microphone()

trigger_word = "패트롤"  # 웨이크 워드
listening = False  # 명령 대기 상태

command_dict = {
                    "앞으로 이동": "FORWARD", 
                    "뒤로 이동": "BACKWㅁRD",
                    "왼쪽으로 이동": "LEFT_MOVE",
                    "오른쪽으로 이동": "RIGHT_MOVE",
                    "왼쪽으로 돌아": "LEFT_TURN",
                    "오른쪽으로 돌아": "RIGHT_TURN",
                    "정지": "STOP"
                }

def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
            sock.sendall(command.encode('utf-8'))
            print(f"명령어 '{command}' 전송 완료.")
    except Exception as e:
        print(f"명령어 전송 실패: {e}")


tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)  # 말하는 속도 조정 (기본값 200)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


while True:
    with mic as source:
        print("음성을 듣는 중...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="ko-KR")  # 한국어 설정
        print(f"인식된 단어: {text}")

        if trigger_word == text and not listening:
            print(f"'{trigger_word}' 감지됨! 다음 음성을 명령으로 인식합니다.")
            speak("네")
            listening = True 
            continue 

        if listening:
            print(f"명령어 전송: {text}")
            if text in command_dict:
                command = command_dict[text]
                print(f"명령어 전송: {command}")
                send_command(command)
            else:
                print(f"알 수 없는 명령어: '{text}'")
                speak("알 수 없는 명령어입니다.")
            listening = False

    except sr.UnknownValueError:
        print("음성을 인식할 수 없습니다.")
    except sr.RequestError:
        print("음성 인식 서비스 오류 발생.")