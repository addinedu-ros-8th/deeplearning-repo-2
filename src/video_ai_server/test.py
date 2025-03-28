import cv2
import torch
from tqdm import tqdm
from utils import load_model, transform
from get_loader import get_loader
from train import get_model
from PIL import Image
import datetime
from ultralytics import YOLO

def draw_roi(frame):
    roi_x1, roi_y1, roi_x2, roi_y2 = 180, 200, 460, 400
    
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
    
    return [roi_x1, roi_y1, roi_x2, roi_y2]

def get_object_position(frame, yolo_model):
    # YOLO로 객체 감지
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]
    
    if len(boxes) > 0:
        # 첫 번째 객체 사용 (다중 객체는 나중에 확장 가능)
        box = boxes[0].astype(int)
        obj_center_x = (box[0] + box[2]) // 2  # 객체 중심 x 좌표
        obj_center_y = (box[1] + box[3]) // 2  # 객체 중심 y 좌표
        return box, obj_center_x, obj_center_y
    
    return None, None, None

def determine_rotation(frame, roi, obj_box, obj_center_x, obj_center_y):
    if obj_box is None:
        return None
    
    # ROI 중심 계산
    roi_center_x = (roi[0] + roi[2]) // 2
    
    # 객체와 ROI 중심 비교
    if obj_center_x < roi_center_x - 50:  # 왼쪽 (50은 허용 오차)
        return "RIGHT_TURN"
    elif obj_center_x > roi_center_x + 50:  # 오른쪽
        return "LEFT_TURN"
    else:
        return "STOP"

def test(
        num_workers,
        batch_size,
        val_ratio,
        test_ratio,
        model_arch,
        mode,
        dataset,
        beam_width,
        checkpoint_dir,
        model_config,
):
    
    _, _, _, train_dataset, _, _ = get_loader(
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size,
        mode=mode,
        model_arch=model_arch,
        dataset=dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    vocab_size = len(train_dataset.vocab)
    print("Vocabulary size:", vocab_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(model_config, vocab_size, device)
    YOLO_model = YOLO('yolov8n.pt')
    print("Model initialized")
    

    load_model(torch.load(checkpoint_dir, weights_only=True, map_location=device), model)

    print("Starting test...")
    model.eval()
    
    rec_flag = False
    out = None
    file_name = f"/home/pepsi/dev_ws/deeplearning-repo-2/src/admin_pc/video_out/emergency_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    file_size = (640, 480)
    out_image_list = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
    
        with torch.no_grad():
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img).unsqueeze(0).to(device)
            generated_captions_greedy = model.caption_images_beam_search(img, train_dataset.vocab, beam_width=beam_width, mode=mode)
            print(f"Predicted: {generated_captions_greedy[0]}")

            if "걷고" not in generated_captions_greedy[0]:
                if not rec_flag:
                    print("Recording started...")
                    rec_flag = True
                    out = cv2.VideoWriter(file_name, fourcc, fps, file_size)

                cmmand_frame = frame.copy()

                roi = draw_roi(cmmand_frame)
                
                obj_box, obj_center_x, obj_center_y = get_object_position(cmmand_frame, YOLO_model)
                
                command = determine_rotation(cmmand_frame, roi, obj_box, obj_center_x, obj_center_y)
                
                if obj_box is not None:
                    cv2.rectangle(cmmand_frame, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), (0, 255, 0), 2)
                    cv2.circle(cmmand_frame, (obj_center_x, obj_center_y), 5, (255, 0, 0), -1)

                cv2.putText(cmmand_frame, command, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                out_image_list.append(frame)
                out.write(frame)
            else:
                if rec_flag:
                    print("Recording stopped due to walking detected.")
                    rec_flag = False
                    out.release()
                    out = None
                    out_image_list = []

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
       
if __name__ == "__main__":
    model_config = {}
    model_config['vitcnn_embed_size'] = 256
    model_config['vitcnn_num_layers'] = 1
    model_config['vitcnn_num_heads'] = 4
    
    print("model_config: ", model_config)
    test(
        num_workers=2,
        batch_size=64,
        val_ratio=0.1,
        test_ratio=0.05,
        model_arch="vitcnn-attn",
        mode="image",
        dataset="imgCaptions",
        beam_width=3,
        checkpoint_dir="/home/pepsi/dev_ws/deeplearning-repo-2/src/video_ai_server/checkpoints/vitcnn-attn/imgCaptions/bs64_lr0.0005_es256_nl1/checkpoint_epoch_50.pth.tar",
        model_config=model_config,
    )