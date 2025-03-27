import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

class Vocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.vocab_size = 4

    def add_word(self, word):
        if word not in self.stoi:
            idx = len(self.stoi)
            self.stoi[word] = idx
            self.itos[idx] = word
            self.vocab_size += 1

    def build_vocab(self, captions):
        for caption in captions:
            for word in caption.split():
                self.add_word(word)

    def numericalize(self, text):
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in text.split()]

class VideoCaptionDataset(Dataset):
    def __init__(self, video_files, captions, vocab, transform=None, frame_interval=30):
        self.video_files = video_files
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        self.frame_interval = frame_interval

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (299, 299))  # inception_v3에 맞게 299x299로 변경
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            frame_count += 1
        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        frames = torch.stack(frames)

        caption = self.captions[idx]
        numerical_caption = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
        caption_tensor = torch.tensor(numerical_caption, dtype=torch.long)

        return frames, caption_tensor

# 커스텀 collate 함수
def custom_collate_fn(batch):
    frames_list = [item[0] for item in batch]
    captions = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    return frames_list, captions