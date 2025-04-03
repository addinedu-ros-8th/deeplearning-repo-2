# deeplearning-repo-2
딥러닝 프로젝트 2조 저장소.


# 정찰 로봇 프로젝트 (Patrol Robot)


<img src="https://github.com/user-attachments/assets/ca0d6ecd-b237-46ef-bb97-2d98343fc26c" width="600" height="500"/>



## 프로젝트 소개
> 외부/내부 장소를 가리지 않고 사람이 직접 가지 않더라도, 자율주행 로봇이 순찰을 하며 위험 상황을 인식하고 이를 알릴 수 있는 시스템을 개발하는 것이 목표입니다.

이 로봇은 장애물을 회피하며 경로를 따라 자율주행하고, 영상/음성/딥러닝 기반 기술을 활용해 사람을 인식하고 위험 상황(싸움, 화재, 쓰러짐 등)을 판단합니다. 녹화된 영상 데이터는 추후 관리자에게 제공되며, 긴급 상황시 알림도 가능합니다.

---

## 기술 스택

| 분류                | 사용 기술                          |
|---------------------|------------------------------------|
| **Language**        | ![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![MySQL](https://img.shields.io/badge/MySQL-Database-blue) |
| **GUI Framework**   | ![PyQt5](https://img.shields.io/badge/PyQt5-GUI_Framework-lightgrey) ![Tkinter](https://img.shields.io/badge/Tkinter-GUI_Framework-lightgrey) |
| **영상/얼굴 인식**  | ![OpenCV](https://img.shields.io/badge/OpenCV-Video_Processing-orange) ![Mediapipe](https://img.shields.io/badge/Mediapipe-Face_Landmarks-red) |
| **음성 처리**       | ![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-Audio-yellow) |
| **딥러닝 프레임워크**| ![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-blueviolet) ![YOLOPose](https://img.shields.io/badge/YOLO_Pose-Pose_Estimation-lightblue) ![BOTSort](https://img.shields.io/badge/BOT_SORT-Multi_Object_Tracking-green) ![LSTM](https://img.shields.io/badge/LSTM-Sequence_Modeling-orange) |
| **자연어 처리**     | ![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow) |

---

## 팀 소개

| 이름     |역할|                                            
|---|---|                        
| 송원준(팀장)|위험 상황 탐지 및 알림 알고리즘<br>위험 상황 탐지 모델 학습<br>위험 상황 탐지 모델 추론<br>음성인식 모델|                        
| 이명운   |장애물 탐지 및 회피 알고리즘<br>장애물 탐지 모델<br>depth 추정 모델|                  
| 심재헌   |TCP 통신<br>주행 알고리즘<br>주행 구현|                     
| 이정림   |UDP 통신<br>GUI<br>발표 자료|

---

## 주제: 정찰 로봇

### 정찰 로봇 :
- 인력이 부족해 사람이 직접 가기 어려운 장소에 대신 들어가 위험 상황을 감지하고 순찰하는 로봇
- 장애물 회피 및 경로 기반 자율주행
- 사람 인식 및 위험 상황 판단 가능 (넘어짐, 싸움, 화재 등)
- 영상 기록 및 추후 확인 가능
- 수동 모드로 특정 지역 집중 확인 가능

---


## System Requirements


| ID| Functions| Description |
|---|---|---|
| SR-01  | 장애물 인식 기능| - 동적 장애물 : 움직이는 사람<br>- 정적 장애물 : 서 있는 사람, 소화기|
| SR-02  | 장애물 회피 기능        | - 정적 장애물의 경우 주변에 장애물이 더 있는지 확인 후 없는 방향으로 우회한다.<br>- 동적 장애물의 경우 진행 방향을 예측하여 반대 방향으로 우회한다. |
| SR-03  | 위험 상황 판단 기능     | - 쓰러짐<br>- 몸 싸움<br>- 화재<br>- 울음|
| SR-04  | 위험 상황 시 녹화 기능  | - 위험 상황을 파악했을 시 녹화 시작|
| SR-05  | 음성 인식              | - 목적지 설정 명령 인식(방향 설정)|
| SR-06  | 수동제어 명령어        | - front<br>- back<br>- stop<br>- Left side `<왼쪽 옆으로 단위 길이만큼 이동>`<br>- right side `<오른쪽 옆으로 단위 길이만큼 이동>`<br>- turn right `<오른쪽으로 단위 각도만큼 회전>`<br>- turn left `<왼쪽으로 단위 각도만큼 회전>` |
| SR-07  | 수동 제어| - 키보드로 입력 받은 명령에 따라 제어되는 기능<br>- 음성입력으로 입력 받은 명령에 따라 제어되는 기능|
| SR-08  | 인터페이스             | - 모드 설정<br>  - Patrol<br>    • 예약으로 Patrol 시작<br>    • 수동으로 Patrol 시작<br>  - Manual<br>    • 위험 상황 감지, 장애물 감지 로그 확인 기능<br>    • 위험 상황 시 녹화 된 영상 재생<br>    • 위험 상황 발생률 그래프 |


---


## System Architecture
<img src="https://github.com/user-attachments/assets/353dd495-0ca7-496e-bb25-2770ea760f34">


---

## Data Structure

- ER Diagram
 <img src="https://github.com/user-attachments/assets/b33ecdfe-df58-4a12-8c35-fcd07221bd28" width="700" height="500"/>                            


  
- Class Diagram
 <img src="https://github.com/user-attachments/assets/22214890-d8c7-4940-8ddd-7f014726aee7" width="700" height="500"/>                                         
 <img src="https://github.com/user-attachments/assets/4f168637-d9a3-4522-9fb7-d8b6aaf43d55" width="400" height="300"/>                         
 

---

## GUI Wireframe
- LogIn GUI             
 <img src="https://github.com/user-attachments/assets/4a802fbf-77fe-4b7e-80d9-05f64c663a48" width="1000" height="400"/>                             
- Main GUI                 
 <img src="https://github.com/user-attachments/assets/79020cf1-afbd-4842-8a2c-4f3daafa3e28" width="1000" height="400"/>                 
- Log GUI              
 <img src="https://github.com/user-attachments/assets/67e398f9-8695-4d8e-9101-24ce90dc0a17" width="1000" height="400"/>                 
- Graph GUI                
 <img src="https://github.com/user-attachments/assets/a9449510-115b-4713-84f4-76a01a3ae6de" width="1000" height="400"/>                 

---

## State Diagram

<img src="https://github.com/user-attachments/assets/a8c650a5-1a71-4de3-a61d-1a3948407a96"/>


---


## Sequence Diagram

- 순찰 Success                    
 <img src="https://github.com/user-attachments/assets/79c6911c-cd7f-4173-ba45-846f94429901"/>                  

- 순찰 중 위험 감지                          
 <img src="https://github.com/user-attachments/assets/a00d0aba-f375-4d90-8cfa-51c677c91a6d"/>                         

- 순찰 종료                       
 <img src="https://github.com/user-attachments/assets/2d39ac0b-14ce-42cb-a78b-abd55a8a149a"/>                 

- 장애물 회피                              
 <img src="https://github.com/user-attachments/assets/09a44e2a-8ab5-4de8-9814-98024216b9c3"/>                       

- 수동 제어                   
 <img src="https://github.com/user-attachments/assets/198f8f7c-9b46-449f-94e4-b53175c8ec6b"/>                          

- 수동 제어 실패                
 <img src="https://github.com/user-attachments/assets/df3eb515-45db-4d66-8b23-7f29f3d19f1c"/>



