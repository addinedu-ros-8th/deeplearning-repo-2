import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm

class DataPreprocessing:
    def __init__(self, video_dir='./datasets/pose/', output_dir='./datasets/pose/imgCaptions/', frame_rate=5):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.captions = {}
    def extract_images(self):
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        video_paths = glob.glob(os.path.join(self.video_dir, "*.mp4"))
        for video_path in tqdm(video_paths, desc="Processing Videos"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cap = cv2.VideoCapture(video_path)
            fps = int(cv2.CAP_PROP_FPS)

            frame_interval = fps // 5
            frame_count = 0
            extracted_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    file_name = f"{video_name}_{extracted_count}.jpg"
                    file_path = os.path.join(self.output_dir, "images", file_name)
                    cv2.imwrite(file_path, frame)

                    self.captions[file_name] = video_name
                    extracted_count += 1

                frame_count += 1

            cap.release()

    def write_csv(self):
        df = pd.DataFrame(list(self.captions.items()), columns=["image", "caption"])
        df.to_csv(os.path.join(self.output_dir, 'captions.txt'), index=False)
        print(f"모든 비디오에서 {len(self.captions)}개 이미지 및 라벨 저장 완료")

if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.extract_images()
    dp.write_csv()