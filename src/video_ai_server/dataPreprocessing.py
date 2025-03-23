import cv2
import os
import glob
import json
from tqdm import tqdm

class DataPreprocessing:
    def __init__(self, video_dir='./datasets/pose/', output_dir='./datasets/pose/imgCaptions/', frame_interval=10):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.captions = {}

    def make_dir(self):
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)

    def extract_images(self):
        video_paths = glob.glob(os.path.join(self.video_dir, "*.mp4"))

        for video_path in tqdm(video_paths, desc="Processing Videos"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cap = cv2.VideoCapture(video_path)

            frame_count = 0
            img_count = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_count % self.frame_interval == 0:
                    file_name = f"{video_name}_frame_{img_count}.jpg"
                    file_path = os.path.join(self.output_dir, "images", file_name)
                    cv2.imwrite(file_path, frame)

                    self.captions[file_name] = video_name
                    img_count += 1
                
                frame_count += 1

            cap.release()

    def write_json(self):
        json_path = os.path.join(self.output_dir, "labels", "captions.json")
        with open(json_path, 'w') as f:
            json.dump(self.captions, f, indent=4)

        print(f"모든 비디오에서 {len(self.captions)}개 이미지 및 라벨 저장 완료")

if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.make_dir()
    dp.extract_images()
    dp.write_json()
