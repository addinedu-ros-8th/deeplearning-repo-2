import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string
import pickle
seed = 42

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.tokenizer = word_tokenize

    def __len__(self):
        return(self.itos)

    def tokenizer_kor(self, text):
        if not text or pd.isnull(text):
            return []
        
        text = text.strip().strip("\n")
        text = "".join([char for char in text if char not in string.punctuation])
        return [tok for tok in self.tokenizer(text)][:48]
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentences in sentence_list:
            for sentence in sentences:
                for word in self.tokenizer_kor(sentence):
                    if word not in frequencies:
                        frequencies[word] = 1

                    else:
                        frequencies[word] += 1

                    if frequencies[word] == self.freq_threshold:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_kor(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, mode='precomputed', precomputed_dir=None, dataset=None, model_arch=None, transform=None, freq_threshold=5, max_length=70):
        self.root_dir = root_dir
        self.df = captions_file
        self.transform = transform
        
        self.mode = mode
        self.precomputed_dir = precomputed_dir
        self.dataset = dataset
        self.model_arch = model_arch
        
        self.df = self.df.dropna()

        self.imgs = self.df["image"].tolist()
        raw_captions = self.df["caption"].tolist()

        for i in range(len(raw_captions)):
            for j in range(len(raw_captions[i])):
                if pd.isnull(raw_captions[i][j]):
                    raw_captions[i][j] = ""
                else:
                    raw_captions[i][j] = raw_captions[i][j].lower().strip().strip("\n")
                    raw_captions[i][j] = "".join([char for char in raw_captions[i][j] if char not in string.punctuation])

        self.ref_captions = raw_captions
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(raw_captions)

        self.numericalized_captions = []
        for captions in raw_captions:
            caption_list = []
            for caption in captions:
                tokens = self.vocab.tokenizer_eng(caption)
                
                numericalized = [self.vocab.stoi["<SOS>"]]
                numericalized += self.vocab.numericalize(caption)
                numericalized.append(self.vocab.stoi["<EOS>"])
                caption_list.append(numericalized)
            
            self.numericalized_captions.append(caption_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ref_caption = self.ref_captions[index]
    
        numericalized_caption = self.numericalized_captions[index]
        
        
        length = len(numericalized_caption)
    
        caption_idx = torch.randint(0, length, (1,)).item()
        numericalized_caption = numericalized_caption[caption_idx]
        
        img_id = self.imgs[index]
        if self.mode == 'image':
            img = Image.open(os.path.join(self.root_dir, str(img_id)))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if self.transform is not None:
                img = self.transform(img)

        elif self.mode == 'precomputed':
            with open(os.path.join(self.precomputed_dir, self.model_arch, self.dataset, str(img_id).split(".")[0] + ".pkl"), 'rb') as f:
                img = pickle.load(f)
                
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)

        return img_id, img, numericalized_caption, ref_caption

class MyCollate:
    def __init__(self, pad_idx, mode='precomputed'):
        self.pad_idx = pad_idx
        self.mode = mode

    def __call__(self, batch):
        img_ids = [item[0] for item in batch]
        targets = [torch.tensor(item[2]) for item in batch]
    
        ref_caption = [item[3] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
            
        if self.mode == 'precomputed':
            precomputed_outputs = [item[1].unsqueeze(0) for item in batch]
            precomputed_outputs = torch.cat(precomputed_outputs, dim=0)
            return img_ids, precomputed_outputs, targets, ref_caption
            
        elif self.mode == 'image':
            imgs = [item[1].unsqueeze(0) for item in batch]
            imgs = torch.cat(imgs, dim=0)
            return img_ids, imgs, targets, ref_caption
        
        else:
            raise ValueError("Invalid mode. Choose either 'precomputed' or 'image'")

def get_loader(
    transform,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size=32,
    num_workers=8,
    mode='precomputed',
    dataset='mscoco',
    model_arch='cnn-rnn',
    shuffle=True,
    pin_memory=True,
):
    precomputed_dir = './precomputed/'
    
    root_folder = "./datasets/pose/images/"
    captions_path = "./datasets/pose/captions.txt"
    
    img_captions = pd.read_csv(captions_path)
    img_captions = img_captions.groupby("image").agg(list).reset_index()
    
    train_val_img_captions, test_img_captions = train_test_split(
        img_captions, test_size=test_ratio, random_state=seed
    )
    train_img_captions, val_img_captions = train_test_split(
        train_val_img_captions, test_size=val_ratio, random_state=seed
    )
    
    #print train, val, test, size
    print("Train size: ", len(train_img_captions))
    print("Val size: ", len(val_img_captions))
    print("Test size: ", len(test_img_captions))
    
    train_dataset = ImageCaptionDataset(root_folder, train_img_captions, mode=mode, precomputed_dir=precomputed_dir, dataset=dataset, model_arch=model_arch, transform=transform)
    val_dataset = ImageCaptionDataset(root_folder, val_img_captions, mode=mode, precomputed_dir=precomputed_dir, dataset=dataset, model_arch=model_arch, transform=transform)
    test_dataset = ImageCaptionDataset(root_folder, test_img_captions, mode=mode, precomputed_dir=precomputed_dir, dataset=dataset, model_arch=model_arch, transform=transform)

    
    pad_idx = train_dataset.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx, mode=mode),
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx, mode=mode),
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx, mode=mode),
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset