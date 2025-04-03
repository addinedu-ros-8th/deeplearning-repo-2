import cv2
import socket
import numpy as np
import threading
from ultralytics import YOLO
import os

# UDP 설정
udp_ip = "0.0.0.0"  # 모든 네트워크 인터페이스에서 수신
udp_port1 = 6000   # 첫 번째 카메라 포트 (왼쪽)
udp_port2 = 7000   # 두 번째 카메라 포트 (오른쪽)
MAX_PACKET_SIZE = 60000

# 데이터 버퍼 및 프레임 저장소
buffers = {"CAM1": {}, "CAM2": {}}
frames = {"CAM1": None, "CAM2": None}
lock = threading.Lock()

# UDP 수신 함수
def receive_video(udp_port, cam_id):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
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

# 스테레오 캘리브레이션 함수
def compute_depth_physical(disparity, focal_length_px, baseline):
    if disparity <= 0:
        return -1
    return (focal_length_px * baseline) / disparity

def calibrate_stereo():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*6, 3), np.float32)
    square_size = 0.021
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size
    objpoints, imgpointsR, imgpointsL = [], [], []
    print('Starting calibration...')
    for i in range(0, 67):
        ChessImaR = cv2.imread(f'../chessboard-R{i}.png', 0)
        ChessImaL = cv2.imread(f'../chessboard-L{i}.png', 0)
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

# 서버 소켓 설정
SERVER_HOST = '192.168.65.239'
SERVER_PORT = 6001
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_HOST, SERVER_PORT))
print(f"서버 {SERVER_HOST}:{SERVER_PORT}에 연결")

# 캘리브레이션 데이터 로드
calib_data = load_calibration()
mtxL, distL, mtxR, distR, R, T, RL, RR, PL, PR, Left_Stereo_Map0, Left_Stereo_Map1, Right_Stereo_Map0, Right_Stereo_Map1, focal_length_px, baseline_m = calib_data
Left_Stereo_Map = (Left_Stereo_Map0, Left_Stereo_Map1)
Right_Stereo_Map = (Right_Stereo_Map0, Right_Stereo_Map1)

# 스테레오 매칭 설정
window_size, min_disp, num_disp = 5, 2, 128
stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=32, disp12MaxDiff=5,
                               P1=8 * 3 * window_size**2, P2=32 * 3 * window_size**2)
stereoR = cv2.ximgproc.createRightMatcher(stereo)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.8)

# YOLO 모델 로드
model = YOLO('../model/fire_best.pt')

# 스레드 시작
thread1 = threading.Thread(target=receive_video, args=(udp_port1, "CAM1"), daemon=True)
thread2 = threading.Thread(target=receive_video, args=(udp_port2, "CAM2"), daemon=True)
thread1.start()
thread2.start()

# 상태 관리 변수
roi_x1, roi_y1, roi_x2, roi_y2 = 180, 200, 480, 400
roi_center = (roi_x1 + roi_x2) // 2
filteredImg_prev, prev_action, current_action = None, None, None

# 메인 루프
while True:
    with lock:
        frameL = frames["CAM1"]  # 좌측 카메라
        frameR = frames["CAM2"]  # 우측 카메라

    if frameL is None or frameR is None:
        continue  # 프레임이 준비되지 않았으면 건너뜀

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

    threats = []
    for box in boxes:
        track_id = box.id
        if track_id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        in_roi = (x1 < roi_x2 and x2 > roi_x1 and y1 < roi_y2 and y2 > roi_y1)
        if not in_roi:
            continue
        region = disp_map[cy - 2:cy + 3, cx - 2:cx + 3]
        vals = region[region > 0]
        if vals.size == 0:
            continue
        avg_disp = np.mean(vals)
        distance = compute_depth_physical(avg_disp, focal_length_px, baseline_m)
        if distance <= 0 or distance > 0.45:
            continue
        threats.append((distance, cx, track_id))
        label = f"ID:{track_id.item()} {float(distance):.2f}m"
        cv2.rectangle(Left_nice, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(Left_nice, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if threats:
        threats.sort()
        _, cx, _ = threats[0]
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
