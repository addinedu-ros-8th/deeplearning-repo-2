import os
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

class DataPreprocessing:
    def __init__(self, video_dir='/home/pepsi/Downloads/238-2.실내(편의점, 매장) 사람 이상행동 데이터/01-1.정식개방데이터/Training/01.원천데이터/', xml_dir='/home/pepsi/Downloads/238-2.실내(편의점, 매장) 사람 이상행동 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/', output_dir='/home/pepsi/dev_ws/deeplearning-repo-2/src/video_ai_server/datasets/pose/imgCaptions/', frame_rate=5):
        self.video_dir = video_dir
        self.xml_dir = xml_dir
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.captions = {}

        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

    def parse_action_ranges(self, xml_path):
        """XML에서 행동 구간 (start~end 프레임 범위) 추출"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        actions = {
            "fight": ("fight_start", "fight_end"),
            "fall": ("fall_start", "fall_end"),
            "smoke": ("smoke_start", "smoke_end"),
        }

        action_ranges = {}

        for action, (start_label, end_label) in actions.items():
            start_frame, end_frame = None, None
            for track in root.findall("track"):
                label = track.attrib.get("label")
                if label == start_label:
                    start_frame = int(track.find("box").attrib.get("frame"))
                elif label == end_label:
                    end_frame = int(track.find("box").attrib.get("frame"))

            if start_frame is not None and end_frame is not None:
                action_ranges[action] = set(range(start_frame, end_frame + 1))

        return action_ranges

    def get_caption_for_frame(self, frame_idx, action_ranges):
        """프레임 번호에 따라 적절한 캡션 반환"""
        if frame_idx in action_ranges.get("fight", set()):
            return "두 사람이 싸우고 있습니다."
        elif frame_idx in action_ranges.get("fall", set()):
            return "한 사람이 쓰러져 있습니다."
        elif frame_idx in action_ranges.get("smoke", set()):
            return "한 사람이 담배를 피우고 있습니다."
        else:
            return "사람들이 정상적으로 움직이고 있습니다."

    def extract_images(self):
        """비디오를 프레임으로 추출하고 캡션을 부여"""
        for label_folder in os.listdir(self.video_dir):
            video_folder = os.path.join(self.video_dir, label_folder)
            video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))

            for video_path in tqdm(video_paths, desc=f"Processing {label_folder}"):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                cap = cv2.VideoCapture(video_path)

                # XML 경로 및 행동 범위 파싱
                xml_path = os.path.join(self.xml_dir, label_folder, f"{video_name}.xml")
                action_ranges = self.parse_action_ranges(xml_path) if os.path.exists(xml_path) else {}

                frame_idx = 0
                saved_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % self.frame_rate == 0:
                        image_name = f"{video_name}_{saved_count:04d}.jpg"
                        image_path = os.path.join(self.output_dir, "images", image_name)
                        cv2.imwrite(image_path, frame)

                        caption = self.get_caption_for_frame(frame_idx, action_ranges)
                        self.captions[image_name] = caption
                        saved_count += 1

                    frame_idx += 1

                cap.release()

    def write_csv(self):
        """Flickr30k 포맷 캡션 CSV 저장"""
        df = pd.DataFrame(list(self.captions.items()), columns=["image", "caption"])
        df.to_csv(os.path.join(self.output_dir, 'captions.txt'), index=False, encoding="utf-8-sig")
        print(f"\n✅ 총 {len(self.captions)}개의 이미지 및 캡션 저장 완료!")

if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.extract_images()
    dp.write_csv()
