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
cnt = 0


############################################
# 전역 변수로 뎁스 맵과 객체 정보 공유
############################################
shared_data = {
    "Left_nice": None,
    "filtered": None,
    "objects": [],
}

############################################
# UDP 수신 함수
############################################
def receive_video(udp_port, cam_id):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, udp_port))
    while True:
        try:
            data, addr = sock.recvfrom(MAX_PACKET_SIZE + 50)  # 헤더 포함 크기
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
# depth 계산
############################################
def compute_depth_physical(disparity, focal_length_px, baseline):
    if isinstance(disparity, np.ndarray):
        disparity = disparity.item()  # 배열 -> 스칼라 변환
    if disparity <= 0:
        return -1
    return float((focal_length_px * baseline) / disparity)


############################################
# 뎁스 계산 및 객체 감지 함수
############################################
def compute_depth_and_objects(frameL, frameR, Left_Stereo_Map, Right_Stereo_Map, stereo, stereoR, wls_filter, model, focal_length_px, baseline_m):
    Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4)
    Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4)
    grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(grayL, grayR).astype(np.int16)
    dispR = stereoR.compute(grayR, grayL).astype(np.int16)
    filtered = wls_filter.filter(disp, grayL, None, dispR)
    disp_map = filtered.astype(np.float32) / 16.0

    results = model.track(Left_nice, conf=0.6, persist=True, verbose=False)
    boxes = results[0].boxes if len(results) > 0 else []

    objects = []
    for box in boxes:
        track_id = box.id
        if track_id is None:
            continue
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        region = disp_map[cy - 3:cy + 4, cx - 3:cx + 4]
        vals = region[region > 0]
        if vals.size == 0:
            continue
        p_low, p_high = np.percentile(vals, [20, 90])
        vals_filtered = vals[(vals >= p_low) & (vals <= p_high)]
        if vals_filtered.size == 0:
            continue
        avg_disp = np.mean(vals_filtered)
        distance = compute_depth_physical(avg_disp, focal_length_px, baseline_m)
        if distance <= 0:
            continue

        objects.append({
            "track_id": int(track_id),
            "class_name": class_name,
            "distance": float(distance),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy
        })

    return Left_nice, filtered, objects


############################################
# 행동분석 함수
############################################
def motionPrediction(class_name, distance):
    if class_name in ["fire", "person"]:  # 불과 사람은 danger
        return "danger"
    if class_name == "fire_extinguisher":  # 소화기는 normal (회피)
        return "normal"
    return "normal"


############################################
# 캘리브레이션 및 저장
############################################
def calibrate_stereo():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9 * 6, 3), np.float32)
    square_size = 0.021
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size
    objpoints, imgpointsR, imgpointsL = [], [], []

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
        criteria=criteria_stereo, flags=cv2.CALIB_FIX_INTRINSIC)
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, alpha=0)
    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)
    focal_length_px = mtxL[0, 0]
    baseline_m = abs(T[0])
    np.savez('stereo_calibration.npz', mtxL=MLS, distL=dLS, mtxR=MRS, distR=dRS, R=R, T=T, RL=RL, RR=RR, PL=PL, PR=PR,
             Left_Stereo_Map0=Left_Stereo_Map[0], Left_Stereo_Map1=Left_Stereo_Map[1],
             Right_Stereo_Map0=Right_Stereo_Map[0], Right_Stereo_Map1=Right_Stereo_Map[1],
             focal_length_px=focal_length_px, baseline_m=baseline_m)
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
    if emergency_mode:
        return
    emergency_mode = True

    client_socket.send(stop_action.encode('utf-8'))
    print("Emergency STOP sent. 녹화시작")

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('emergency_recording.avi', fourcc, 20.0, (640, 480))  # 해상도 조정 필요

    while True:
        with lock:
            Left_nice = shared_data["Left_nice"]
            filtered = shared_data["filtered"]
            objects = shared_data["objects"].copy()

        if Left_nice is None or filtered is None:
            time.sleep(0.1)
            continue

        fire_or_person_detected = False
        for obj in objects:
            ret = motionPrediction(obj["class_name"], obj["distance"])
            if ret == "danger":  # fire 또는 person 감지
                fire_or_person_detected = True

            label = f"ID:{obj['track_id']} {obj['distance']:.2f}m {ret}"
            cv2.rectangle(Left_nice, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), (0, 255, 0), 2)
            cv2.putText(Left_nice, label, (obj["x1"], obj["y1"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        #out.write(Left_nice)
        filteredImg = np.clip(filtered, 0, 128 * 16).astype(np.float32)
        filteredImg = (filteredImg / (128 * 16)) * 255.0
        filteredImg = np.uint8(filteredImg)
        disp_color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)
        cv2.imshow("YOLO + Depth (Emergency)", Left_nice)
        cv2.imshow("Filtered Depth (Emergency)", disp_color)

        if not fire_or_person_detected:  # fire와 person이 모두 사라짐
            print("Emergency condition cleared. 녹화종료")
            break

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    #out.release()
    resume_command = prev_action if prev_action is not None else "FORWARD"
    client_socket.send(resume_command.encode('utf-8'))
    print("Resuming operation with command:", resume_command)
    emergency_mode = False

############################################
# 뎁스 추정 및 명령 생성
############################################
def start_depth_action():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("서버 연결 대기중")
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    print(f"서버 {SERVER_HOST}:{SERVER_PORT}에 연결완료")

    calib_data = load_calibration()
    mtxL, distL, mtxR, distR, R, T, RL, RR, PL, PR, Left_Stereo_Map0, Left_Stereo_Map1, Right_Stereo_Map0, Right_Stereo_Map1, focal_length_px, baseline_m = calib_data
    Left_Stereo_Map = (Left_Stereo_Map0, Left_Stereo_Map1)
    Right_Stereo_Map = (Right_Stereo_Map0, Right_Stereo_Map1)

    window_size, min_disp, num_disp = 15, 0, 128
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp, 
                                   numDisparities = num_disp, 
                                   blockSize = window_size,
                                   uniquenessRatio = 10, 
                                   speckleWindowSize = 100, 
                                   speckleRange = 48, 
                                   disp12MaxDiff = 5,
                                   P1 = 8 * 3 * window_size ** 2, 
                                   P2 = 32 * 3 * window_size ** 2)
    # noinspection PyUnusedLocal
    stereoR = cv2.ximgproc.createRightMatcher(stereo)  # 경고 억제

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.8)

    model = YOLO('../model/merge_best.pt')

    thread1 = threading.Thread(target=receive_video, args=(UDP_PORT1, "CAM1"), daemon=True)
    thread2 = threading.Thread(target=receive_video, args=(UDP_PORT2, "CAM2"), daemon=True)
    thread1.start()
    thread2.start()

    roi_x1, roi_y1, roi_x2, roi_y2 = 180, 240, 520, 400
    roi_center = (roi_x1 + roi_x2) // 2
    filteredImg_prev, prev_action, current_action = None, None, None
    stop_action = "STOP"

    while True:
        with lock:
            frameL = frames["CAM1"]
            frameR = frames["CAM2"]
        if frameL is None or frameR is None:
            if frameL is None:
                print("None L frame", end=' ')
            if frameR is None:
                print("None R frame", end=' ')   
            print()
            continue

        Left_nice, filtered, objects = compute_depth_and_objects(frameL, frameR, Left_Stereo_Map, Right_Stereo_Map, stereo, stereoR, wls_filter, model, focal_length_px, baseline_m)

        with lock:
            shared_data["Left_nice"] = Left_nice.copy()
            shared_data["filtered"] = filtered.copy()
            shared_data["objects"] = objects.copy()

        threats = []
        for obj in objects:
            label = f"ID:{obj['track_id']} {obj['distance']:.2f}m"
            cv2.rectangle(Left_nice, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), (0, 255, 0), 2)
            cv2.putText(Left_nice, label, (obj["x1"], obj["y1"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if not emergency_mode:
                ret = motionPrediction(obj["class_name"], obj["distance"])
                if ret == "danger":  # fire 또는 person 감지 시 비상 처리
                    threading.Thread(target=handle_emergency, args=(client_socket, stop_action, prev_action), daemon=True).start()
                    break

            # 소화기 회피 로직
            if obj["class_name"] == "extinguisher":
                in_roi = (obj["x1"] < roi_x2 and obj["x2"] > roi_x1 and obj["y1"] < roi_y2 and obj["y2"] > roi_y1)
                if in_roi and obj["distance"] <= 0.34:  # ROI 내 0.4m 이내
                    threats.append((obj["distance"], obj["cx"], obj["track_id"]))

        if not emergency_mode:
            if threats:  # 소화기 감지 시 회피
                threats.sort()
                _, cx, track_id = threats[0]
                current_action = 'RIGHT_MOVE' if cx < roi_center else 'LEFT_MOVE'
            else:
                current_action = 'FORWARD'

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

