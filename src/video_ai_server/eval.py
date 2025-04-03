import os
import json
from pathlib import Path
import torch
import yaml
from tqdm import tqdm
import torch.optim as optim
from utils import load_model, transform
from get_loader import get_loader
from nlgmetricverse import NLGMetricverse, load_metric
from train import precompute_images, get_model, parse_args

def eval(
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
        saved_name,
):
    if os.path.exists(f'./eval/{model_arch}/{dataset}/{saved_name}'):
        exit(f"Model {model_arch}, {saved_name}, dataset {dataset} already evaluated")
    
    _, _, test_loader, train_dataset, _, _ = get_loader(
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
    print("Model initialized")
    
    if mode == 'precomputed' and not os.path.exists(f'precomputed/{model_arch}/{dataset}'):
        image_train_loader, image_val_loader, image_test_loader, _, _, _ = get_loader(
            transform=transform,
            num_workers=num_workers,
            batch_size=batch_size,
            mode='image',
            model_arch=model_arch,
            dataset=dataset,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        precompute_images(
            model,
            model_arch,
            dataset,
            image_train_loader,
            image_val_loader,
            image_test_loader
        )

        del image_train_loader, image_val_loader, image_test_loader

    load_model(torch.load(checkpoint_dir, weights_only=True, map_location=device), model)

    bleu = NLGMetricverse(metrics=load_metric("bleu"))
    meteor = NLGMetricverse(metrics=load_metric("meteor"))
    cider = NLGMetricverse(metrics=load_metric("cider"))

    print("Starting evaluation...")
    model.eval()

    all_pred_tokens_greedy = []
    all_pred_tokens_beam = []
    all_caption_tokens = []
    
    all_greedy_img_ids = []
    all_beam_img_ids = []
    
    with torch.no_grad():
        for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            imgs = imgs.to(device)
            generated_captions_greedy = model.caption_images(imgs, train_dataset.vocab, mode=mode)
            print(f"Predicted (greedy): {generated_captions_greedy[0]}")
            print(f"Target: {ref_captions[0]}")
            
            all_greedy_img_ids.extend(img_ids)
            all_pred_tokens_greedy.extend(generated_captions_greedy)
            all_caption_tokens.extend(ref_captions)

    test_bleu_score_greedy = bleu(
        predictions=all_pred_tokens_greedy,
        references=all_caption_tokens,
        reduce_fn='mean')['bleu']['score']

    test_meteor_score_greedy = meteor(
        predictions=all_pred_tokens_greedy,
        references=all_caption_tokens,
        reduce_fn='mean')['meteor']['score']

    test_cider_score_greedy = cider(
        predictions=all_pred_tokens_greedy,
        references=all_caption_tokens,
        reduce_fn='mean')['cider']['score']

    print("Greedy:")
    print(f"BLEU: {test_bleu_score_greedy:.4f} | METEOR: {test_meteor_score_greedy:.4f} | CIDEr: {test_cider_score_greedy:.4f}")
    
    all_caption_tokens = []
    with torch.no_grad():
        for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            imgs = imgs.to(device)
            generated_captions_beam = model.caption_images_beam_search(imgs, train_dataset.vocab, beam_width, mode=mode)

            print(f"Predicted (beam): {generated_captions_beam[0]}")
            print(f"Target: {ref_captions[0]}")
            
            all_beam_img_ids.extend(img_ids)
            all_pred_tokens_beam.extend(generated_captions_beam)
            all_caption_tokens.extend(ref_captions)

    test_bleu_score_beam = bleu(
        predictions=all_pred_tokens_beam,
        references=all_caption_tokens,
        reduce_fn='mean')['bleu']['score']

    test_meteor_score_beam = meteor(
        predictions=all_pred_tokens_beam,
        references=all_caption_tokens,
        reduce_fn='mean')['meteor']['score']

    test_cider_score_beam = cider(
        predictions=all_pred_tokens_beam,
        references=all_caption_tokens,
        reduce_fn='mean')['cider']['score']
    
    print("beam:")
    print(f"BLEU: {test_bleu_score_beam:.4f} | METEOR: {test_meteor_score_beam:.4f} | CIDEr: {test_cider_score_beam:.4f}")
    
    metrics = {
        'val_greedy_bleus': test_bleu_score_greedy,
        'val_greedy_meteors': test_meteor_score_greedy,
        'val_greedy_ciders': test_cider_score_greedy,
        'val_beam_bleus': test_bleu_score_beam,
        'val_beam_meteors': test_meteor_score_beam,
        'val_beam_ciders': test_cider_score_beam
    }
    
    model_captions = {}
    for i in range(len(all_greedy_img_ids)):
        model_captions[all_greedy_img_ids[i]] = {
            'greedy': all_pred_tokens_greedy[i]
        }
    for i in range(len(all_beam_img_ids)):
        model_captions[all_beam_img_ids[i]]['beam'] = all_pred_tokens_beam[i]
    
    model_captions = dict(sorted(model_captions.items()))
    
    captions_file_path = Path('./eval/captions.json')
    captions_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not captions_file_path.exists():
        captions_file_path.write_text('{}')
    
    all_captions = json.loads(captions_file_path.read_text())
    
    if model_arch not in all_captions:
        all_captions[model_arch] = {}
    if dataset not in all_captions[model_arch]:
        all_captions[model_arch][dataset] = {} 
    
    all_captions[model_arch][dataset][saved_name] = model_captions
    
    with open(captions_file_path, 'w') as json_file:
        json.dump(all_captions, json_file, indent=4)
        
    print(f"Captions successfully saved to {captions_file_path}")
    
    eval_file_path = Path('./eval/metrics.json')
    eval_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not eval_file_path.exists():
        eval_file_path.write_text('{}')
        
    eval_data = json.loads(eval_file_path.read_text())
    
    if model_arch not in eval_data:
        eval_data[model_arch] = {}
    if dataset not in eval_data[model_arch]:
        eval_data[model_arch][dataset] = {}
    
    eval_data[model_arch][dataset][saved_name] = metrics
    
    with open(eval_file_path, 'w') as json_file:
        json.dump(eval_data, json_file, indent=4)

    print(f"Metrics successfully saved to {eval_file_path}")
       
if __name__ == "__main__":
    args = parse_args()
    
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    embed_size = args.embed_size
    num_layers = args.num_layers
    model_arch = args.model_arch
    dataset = args.dataset
    checkpoint_dir = args.checkpoint_dir
    
    print(f"Using checkpoint directory: {checkpoint_dir}")
    model_config = {}
    model_config['vitcnn_embed_size'] = embed_size
    model_config['vitcnn_num_layers'] = num_layers
    model_config['vitcnn_num_heads'] = 4
    
    saved_name = f"bs{batch_size}_lr{learning_rate}_es{embed_size}_nl{num_layers}"

    print(f"Evaluating model {model_arch}, {saved_name}, dataset {dataset}")
    
    print("model_config: ", model_config)
    eval(
        num_workers=2,
        batch_size=batch_size,
        val_ratio=0.1,
        test_ratio=0.05,
        model_arch=model_arch,
        mode="image",
        dataset=dataset,
        beam_width=3,
        checkpoint_dir=checkpoint_dir,
        model_config=model_config,
        saved_name=saved_name,
    )