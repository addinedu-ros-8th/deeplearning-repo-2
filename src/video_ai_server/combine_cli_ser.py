import socket
import threading
import numpy as np
import cv2
import os
from ultralytics import YOLO
import time


############################################
# IP, PORT
############################################
UDP_IP = "0.0.0.0"  # 모든 네트워크 인터페이스에서 수신
UDP_PORT1 = 6000   # 첫 번째 카메라 포트 (왼쪽)
UDP_PORT2 = 7000   # 두 번째 카메라 포트 (오른쪽)
MAX_PACKET_SIZE = 60000
SERVER_HOST = '192.168.28.150'  # 명령 보낼 ip (메인서버쪽)
SERVER_PORT = 6001              # 명령 보낼 포트 (메인서버쪽)

FORWARD_PORT = 5000  # Forward video data to admin GUI's port
FORWARD_IP = "192.168.65.177"  # Forward video data to admin GUI


############################################
# 왼쪽, 오른쪽 프레임 데이터 저장 버퍼
############################################
buffers = {"CAM1": {}, "CAM2": {}}
frames = {"CAM1": None, "CAM2": None}
lock = threading.Lock()
emergency_mode = False


############################################
# UDP 수신 함수
############################################
def receive_video(udp_port, cam_id):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, udp_port))
    while True:
        try:
            data, addr = sock.recvfrom(MAX_PACKET_SIZE + 50)  # 헤더 포함 크기
            # 헤더 파싱
            header, chunk = data.split(b":", 3)[0:3], data.split(b":", 3)[3]
            cam_id_recv, index, total_size = header[0].decode(), int(header[1]), int(header[2])
            with lock:
                if cam_id_recv not in buffers:
                    buffers[cam_id_recv] = {}
                buffers[cam_id_recv][index] = chunk
                if sum(len(v) for v in buffers[cam_id_recv].values()) >= total_size:
                    full_data = b"".join([buffers[cam_id_recv][i] for i in sorted(buffers[cam_id_recv].keys())])
                    frame = cv2.imdecode(np.frombuffer(full_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    frames[cam_id_recv] = frame
                    buffers[cam_id_recv] = {}
        except Exception as e:
            print(f"Error in {cam_id}: {e}")
    sock.close()


############################################
# 행동분석 함수
############################################
def motionPrediction():
    # 예: "fall_detected", "fire_detected", "normal"
    return "fall_detected"


############################################
# depth 계산
############################################
def compute_depth_physical(disparity, focal_length_px, baseline):
    if isinstance(disparity, np.ndarray):
        disparity = disparity.item()  # 배열 -> 스칼라 변환
    if disparity <= 0:
        return -1
    return float((focal_length_px * baseline) / disparity)


############################################
# 캘리브레이션 및 저장
############################################
def calibrate_stereo():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*6, 3), np.float32)
    square_size = 0.021
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size
    objpoints, imgpointsR, imgpointsL = [], [], []

    # 캘리브레이션 매개변수 추출
    print('Starting calibration...')
    for i in range(0, 67):
        ChessImaR = cv2.imread(f'../cail_data/chessboard-R{i}.png', 0)
        ChessImaL = cv2.imread(f'../cail_data/chessboard-L{i}.png', 0)
        if ChessImaR is None or ChessImaL is None:
            continue
        retR, cornersR = cv2.findChessboardCorners(ChessImaR, (9,6), None)
        retL, cornersL = cv2.findChessboardCorners(ChessImaL, (9,6), None)
        if retR and retL:
            objpoints.append(objp)
            cv2.cornerSubPix(ChessImaR, cornersR, (11,11), (-1,-1), criteria)
            cv2.cornerSubPix(ChessImaL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)
    retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)
    retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)
    retS, MLS, dLS, MRS, dRS, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, ChessImaR.shape[::-1],
        criteria = criteria_stereo, flags = cv2.CALIB_FIX_INTRINSIC)
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, alpha=0)
    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)
    focal_length_px = mtxL[0, 0]
    baseline_m = abs(T[0])
    np.savez('stereo_calibration.npz', mtxL = MLS, distL = dLS, mtxR = MRS, distR = dRS, R = R, T = T, RL = RL, RR = RR, PL = PL, PR = PR,
             Left_Stereo_Map0 = Left_Stereo_Map[0], Left_Stereo_Map1 = Left_Stereo_Map[1],
             Right_Stereo_Map0 = Right_Stereo_Map[0], Right_Stereo_Map1 = Right_Stereo_Map[1],
             focal_length_px = focal_length_px, baseline_m = baseline_m)
    return (MLS, dLS, MRS, dRS, R, T, RL, RR, PL, PR, Left_Stereo_Map, Right_Stereo_Map, focal_length_px, baseline_m)


############################################
# 캘리브레이션 파일 로드
############################################
def load_calibration():
    if os.path.exists('stereo_calibration.npz'):
        print('Loading saved calibration data...')
        with np.load('stereo_calibration.npz') as data:
            return tuple(data[k] for k in ['mtxL', 'distL', 'mtxR', 'distR', 'R', 'T', 'RL', 'RR', 'PL', 'PR',
                                           'Left_Stereo_Map0', 'Left_Stereo_Map1', 'Right_Stereo_Map0', 'Right_Stereo_Map1',
                                           'focal_length_px', 'baseline_m'])
    else:
        print('Calibration file not found. Performing calibration...')
        return calibrate_stereo()


############################################
# 비상 처리
############################################
def handle_emergency(client_socket, stop_action, prev_action):
    global emergency_mode
    emergency_mode = True
    client_socket.send(stop_action.encode('utf-8'))
    print("emergency STOP")
    print("녹화시작")

    p_predict = motionPrediction()
    while True:
        c_predict = motionPrediction()
        if p_predict != c_predict:
            print("녹화종료")
            break
        time.sleep(1)

    if prev_action is None:
        client_socket.send(b"FORWARD")
    else:
        client_socket.send(prev_action.encode('utf-8'))
    
    emergency_mode = False


############################################
# 뎁스 추정 및 명령 생성
############################################
def start_depth_action():
    # 서버 소켓 설정 (to main server)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("서버 연결 대기중")
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    print(f"서버 {SERVER_HOST}:{SERVER_PORT}에 연결완료")

    calib_data = load_calibration()
    mtxL, distL, mtxR, distR, R, T, RL, RR, PL, PR, Left_Stereo_Map0, Left_Stereo_Map1, Right_Stereo_Map0, Right_Stereo_Map1, focal_length_px, baseline_m = calib_data
    Left_Stereo_Map = (Left_Stereo_Map0, Left_Stereo_Map1)
    Right_Stereo_Map = (Right_Stereo_Map0, Right_Stereo_Map1)

    # 스테레오 매칭 설정
    window_size, min_disp, num_disp = 5, 2, 128
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp, 
                                    numDisparities = num_disp, 
                                    blockSize = window_size,
                                    uniquenessRatio = 10, 
                                    speckleWindowSize = 100, 
                                    speckleRange = 32, 
                                    disp12MaxDiff = 5,
                                    P1 = 8 * 3 * window_size ** 2, 
                                    P2 = 32 * 3 * window_size ** 2)
    stereoR = cv2.ximgproc.createRightMatcher(stereo)

    # wls 필터 적용 
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.8)

    # YOLO 모델 로드(사람 + 소화기 + 불)
    model = YOLO('../model/merge_best.pt')

    # 스레드 시작
    thread1 = threading.Thread(target = receive_video, args = (UDP_PORT1, "CAM1"), daemon = True)
    thread2 = threading.Thread(target = receive_video, args = (UDP_PORT2, "CAM2"), daemon = True)
    thread1.start()
    thread2.start()

    # 상태 관리 변수
    global emergency_mode
    roi_x1, roi_y1, roi_x2, roi_y2 = 180, 200, 480, 400
    roi_center = (roi_x1 + roi_x2) // 2
    filteredImg_prev, prev_action, current_action = None, None, None
    stop_action = "STOP"

    while True:
        with lock:
            frameL = frames["CAM1"]  # 좌측 카메라
            frameR = frames["CAM2"]  # 우측 카메라
        if frameL is None or frameR is None:
            if frameL is None:
                print("None L frame", end = ' ')
            if frameR is None:
                print("None R frame", end = ' ')   
            print()
            continue  # 프레임이 준비되지 않았으면 건너뜀

        # 보정진행 및 remap
        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4)
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4)

        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        disp = stereo.compute(grayL, grayR).astype(np.int16)
        dispR = stereoR.compute(grayR, grayL).astype(np.int16)
        filtered = wls_filter.filter(disp, grayL, None, dispR)
        disp_map = filtered.astype(np.float32) / 16.0

        results = model.track(Left_nice, conf = 0.6, persist = True, verbose = False)
        boxes = results[0].boxes if len(results) > 0 else []

        threats = []
        for box in boxes:
            track_id = box.id
            if track_id is None:
                continue

            # 클래스명 가져오기
            class_id = int(box.cls[0])
            class_name = model.names[class_id] 

            # 좌표계산
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 객체중심 시차 0이하 오류값 제거
            region = disp_map[cy - 2:cy + 3, cx - 2:cx + 3]
            vals = region[region > 0]
            if vals.size == 0:
                continue

            # 시차 이상치 제거
            p_low, p_high = np.percentile(vals, [10, 90])
            vals_filtered = vals[(vals >= p_low) & (vals <= p_high)]
            if vals_filtered.size == 0:
                continue

            # 사이즈 내 평균시차, 초점거리, 베이스라인으로 distance 계산 
            avg_disp = np.mean(vals_filtered)
            distance = compute_depth_physical(avg_disp, focal_length_px, baseline_m)
            
            if distance <= 0:
                continue

            # 객체 id 및 distance 출력
            label = f"ID:{int(track_id)} {float(distance):.2f}m"
            cv2.rectangle(Left_nice, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(Left_nice, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # 불 or 사람 감지 시
            if class_name in ["fire", "person"] and not emergency_mode:
                ret = motionPrediction()
                if ret != "normal":
                    threading.Thread(target = handle_emergency, args =(client_socket, stop_action, prev_action), daemon = True).start()            

            # roi 범위확인
            in_roi = (x1 < roi_x2 and x2 > roi_x1 and y1 < roi_y2 and y2 > roi_y1)
            if in_roi and distance <= 0.5:
                threats.append((distance, cx, track_id))

        if not emergency_mode:
            if threats:
                threats.sort()
                _, cx, track_id = threats[0]
                current_action = 'RIGHT_MOVE' if cx < roi_center else 'LEFT_MOVE'
            else:
                current_action = 'FORWARD'

            # 비상상황 아니고 이전명령 변경되었을 경우
            if prev_action != current_action:
                print(f"Action: {current_action}")
                prev_action = current_action
                client_socket.send(current_action.encode('utf-8'))
    
        filteredImg = np.clip(filtered, 0, num_disp * 16).astype(np.float32)
        filteredImg = (filteredImg / (num_disp * 16)) * 255.0
        filteredImg = np.uint8(filteredImg)
    
        alpha = 0.6
        if filteredImg_prev is not None:
            filteredImg = cv2.addWeighted(filteredImg, alpha, filteredImg_prev, 1 - alpha, 0)
        filteredImg_prev = filteredImg.copy()
        disp_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)
    
        cv2.rectangle(Left_nice, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.imshow("YOLO + Depth", Left_nice)
        cv2.imshow("Filtered Depth", disp_color)
    
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cv2.destroyAllWindows()
    client_socket.close()
    print("클라이언트 소켓이 닫힘")


############################################
# main
############################################
if __name__ == "__main__":
    try:
        start_depth_action()
    except Exception as e:
        print("\nOpen failed.", e)


