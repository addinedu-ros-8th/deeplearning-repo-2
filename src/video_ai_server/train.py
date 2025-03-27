import os
import json
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pickle
import sys
import argparse
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from utils import save_checkpoint, load_checkpoint, transform
from get_loader import Vocabulary, VideoCaptionDataset, custom_collate_fn
from models.yolocnnattn_model import YOLOCNNAttentionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--dataset", type=str, default="video")
    parser.add_argument("--model_arch", type=str, default="cnn-rnn")
    return parser.parse_args()

def precompute_images(
    model,
    model_arch,
    dataset,
    train_loader,
    val_loader,
    test_loader
):
    print("Precomputing images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    with torch.no_grad():
        for idx, (frames_list, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False):
            
            batch_size = len(frames_list)
            outputs_list = []
            for frames in frames_list:
                frames = frames.to(device)
                outputs = model.precompute_image(frames)
                outputs_list.append(outputs)
            
            if not os.path.exists(f'precomputed/{model_arch}/{dataset}'):
                os.makedirs(f'precomputed/{model_arch}/{dataset}')
                
            for i in range(batch_size):
                filepath = f'precomputed/{model_arch}/{dataset}/train_sample_{idx}_{i}.pkl'
                with open(filepath, 'wb') as f:
                    pickle.dump(outputs_list[i].cpu(), f)

def get_model(model_config, vocab_size, device):
    model_arch = model_config['model_arch']
    
    if model_arch == "yolocnn-attn":
        yolocnn_embed_size = model_config['yolocnn_embed_size']
        yolocnn_num_layers = model_config['yolocnn_num_layers']
        yolocnn_num_heads = model_config['yolocnn_num_heads']
        return YOLOCNNAttentionModel(yolocnn_embed_size, vocab_size, yolocnn_num_heads, yolocnn_num_layers).to(device)
    
    else:
        raise ValueError("Model not recognized")

def train(
    learning_rate,
    num_epochs,
    num_workers,
    batch_size,
    val_ratio,
    test_ratio,
    step_size,
    gamma,
    model_arch,
    mode,
    dataset,
    beam_width,
    save_model,
    load_model,
    checkpoint_dir,
    model_config,
    saved_name,
    save_every,
    eval_every
):
    if os.path.exists(f'./checkpoints/{model_arch}/{dataset}/{saved_name}'):
        print(f"Model {model_arch}, {saved_name}, dataset {dataset} already trained")
        sys.exit(0)

    # 학습 데이터 정의
    video_files = ["/home/pepsi/dev_ws/deeplearning-repo-2/src/video_ai_server/datasets/pose/fainting.mp4", "/home/pepsi/dev_ws/deeplearning-repo-2/src/video_ai_server/datasets/pose/normal.mp4", "/home/pepsi/dev_ws/deeplearning-repo-2/src/video_ai_server/datasets/pose/fainting.mp4", "/home/pepsi/dev_ws/deeplearning-repo-2/src/video_ai_server/datasets/pose/smoking.mp4"]
    captions = [
        "A person is fighting",
        "A person is walking normally",
        "A person is fainting",
        "A person is smoking"
    ]

    # Vocabulary 생성
    vocab = Vocabulary()
    vocab.build_vocab(captions)

    # Transform 정의
    transform_composed = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터셋 및 데이터로더 생성
    train_dataset = VideoCaptionDataset(video_files, captions, vocab, transform=transform_composed, frame_interval=30)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화
    model = get_model(model_config, vocab.vocab_size, device)
    
    # Accelerator 설정
    accelerator = Accelerator()
    model, train_loader = accelerator.prepare(model, train_loader)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # TensorBoard 설정
    writer = SummaryWriter(log_dir=f'runs/{model_arch}/{dataset}/{saved_name}')
    
    # 체크포인트 로드 (필요 시)
    start_epoch = 0
    if load_model and os.path.exists(f'{checkpoint_dir}/{saved_name}'):
        start_epoch = load_checkpoint(f'{checkpoint_dir}/{saved_name}', model, optimizer, accelerator.device)
    
    # 학습 루프
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        
        for idx, (frames_list, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            captions = captions.to(accelerator.device)
            batch_size = len(frames_list)

            # 프레임별 인코딩
            enc_outputs = []
            for frames in frames_list:
                frames = frames.to(accelerator.device)
                enc_output = model.precompute_image(frames)
                enc_outputs.append(enc_output)

            # 최대 프레임 수에 맞춰 패딩
            max_frames = max([enc.size(0) for enc in enc_outputs])
            padded_enc_outputs = torch.zeros(batch_size, max_frames, enc_outputs[0].size(1)).to(accelerator.device)
            for i, enc in enumerate(enc_outputs):
                padded_enc_outputs[i, :enc.size(0), :] = enc

            optimizer.zero_grad()
            outputs = model(padded_enc_outputs, captions[:, :-1], mode="precomputed")
            loss = criterion(outputs.view(-1, vocab.vocab_size), captions[:, 1:].reshape(-1))
            
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        
        # 학습률 스케줄러 업데이트
        scheduler.step()
        
        # 모델 저장
        if save_model and (epoch + 1) % save_every == 0:
            checkpoint_path = f'{checkpoint_dir}/{saved_name}'
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path, accelerator)

    # 학습 완료 후 최종 모델 저장
    if save_model:
        checkpoint_path = f'{checkpoint_dir}/{saved_name}'
        save_checkpoint(model, optimizer, num_epochs, checkpoint_path, accelerator)
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    
    # 설정 파일 로드
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # train 함수 호출
    train(
        learning_rate=args.learning_rate,
        num_epochs=config.get('num_epochs', 10),
        num_workers=config.get('num_workers', 4),
        batch_size=args.batch_size,
        val_ratio=config.get('val_ratio', 0.1),
        test_ratio=config.get('test_ratio', 0.1),
        step_size=config.get('step_size', 7),
        gamma=config.get('gamma', 0.1),
        model_arch=args.model_arch,
        mode=config.get('mode', 'train'),
        dataset=args.dataset,
        beam_width=config.get('beam_width', 3),
        save_model=config.get('save_model', True),
        load_model=config.get('load_model', False),
        checkpoint_dir=args.checkpoint_dir,
        model_config=config['model_config'],
        saved_name=config.get('saved_name', 'model_checkpoint'),
        save_every=config.get('save_every', 5),
        eval_every=config.get('eval_every', 5)
    )