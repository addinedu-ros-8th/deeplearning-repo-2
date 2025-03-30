import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define the models, datasets, and inference types to evaluate
model_list = ["vitcnn-attn"]
dataset_list = ["imgCaptions"]
inference_type_list = ["greedy", "beam"]
num_epoch_eval = 50
n = 10

def create_metric_plots(best_model_metrics):
    # 디렉토리가 없으면 생성
    os.makedirs("./eval/metric_plots", exist_ok=True)

    metric_name_list = ["bleus", "ciders", "meteors", "train_losses"]
    for dataset in best_model_metrics:
        for metric_name in metric_name_list:
            if metric_name == "train_losses":
                fig_train, ax2 = plt.subplots(1, 1, figsize=(10, 5))
                for model in best_model_metrics[dataset]:
                    metrics = best_model_metrics[dataset][model]
                    ax2.plot(metrics["train_losses"], label=model)
                    ax2.set_title(f"Train Loss for {dataset}")
                    ax2.set_xlabel("Epoch")
                    ax2.legend()
                fig_train.savefig(f"./eval/metric_plots/{dataset}_{metric_name}_plot.png")
                plt.close(fig_train)
            else:
                fig_val, ax = plt.subplots(1, 2, figsize=(15, 5))
                for model in best_model_metrics[dataset]:
                    metrics = best_model_metrics[dataset][model]
                    for i, inference_type in enumerate(inference_type_list):
                        key = f"val_{inference_type}_{metric_name}"
                        if key in metrics:
                            x_indices = list(range(len(metrics[key])))
                            x_labels = [(i + 1) * 5 for i in x_indices]
                            ax[i].plot(x_indices, metrics[key], label=model)
                            ax[i].set_title(f"{inference_type} {metric_name.upper()} for {dataset}")
                            ax[i].set_xlabel("Epoch")
                            ax[i].legend()
                            ax[i].set_xticks(x_indices)
                            ax[i].set_xticklabels(x_labels)
                        else:
                            print(f"Warning: {key} not found in metrics for {model}, {dataset}")
                fig_val.savefig(f"./eval/metric_plots/{dataset}_{metric_name}_plot.png")
                plt.close(fig_val)

    print("All metric plots saved successfully.")

def select_n_samples(n, model, dataset):
    with open("./eval/captions.json", 'r') as json_file:
        all_captions = json.load(json_file)
    
    first_model = model
    first_dataset = dataset
    first_trained_model = list(all_captions[first_model][first_dataset].keys())[0]
    img_ids = all_captions[first_model][first_dataset][first_trained_model].keys()
    
    # 사용 가능한 샘플 수와 n 비교
    available_samples = len(img_ids)
    n_adjusted = min(n, available_samples)  # n을 최대 사용 가능 수로 제한
    print(f"Requested {n} samples, but only {available_samples} available. Using {n_adjusted} samples.")
    
    n_sample_img_ids = np.random.choice(list(img_ids), n_adjusted, replace=False)
    n_sample_captions = {} 
    
    for img_id in n_sample_img_ids:
        n_sample_captions[img_id] = {}
        for dataset in dataset_list:
            n_sample_captions[img_id][dataset] = {}
            for inference_type in inference_type_list:
                n_sample_captions[img_id][dataset][inference_type] = {}
                for model in model_list:
                    trained_model = list(all_captions[model][dataset].keys())[0]
                    n_sample_captions[img_id][dataset][inference_type][model] = all_captions[model][dataset][trained_model][img_id][inference_type]
                
    with open(f"./eval/{n_adjusted}_sample_captions.json", 'w') as json_file:
        json.dump(n_sample_captions, json_file, indent=4)

if __name__ == "__main__":
    best_model_metrics = {}
    for model in model_list:
        for dataset in dataset_list:
            metric_folder = f"./metric_logs/{model}/{dataset}"
            if not os.path.exists(metric_folder):
                continue
            trained_models = os.listdir(metric_folder)
            best_model_score = 0
            best_model = ""
            best_metrics = {}
            
            for trained_model in trained_models:
                train_model_path = f"./metric_logs/{model}/{dataset}/{trained_model}/train_val_to_epoch_{num_epoch_eval}.json"
                with open(train_model_path, 'r') as json_file:
                    train_data = json.load(json_file)

                if train_data["val_beam_bleus"][-1] > best_model_score:
                    best_model_score = train_data["val_beam_bleus"][-1]
                    best_model = trained_model
                    best_metrics = train_data
                    
                    
            print(f"Best model for {model} on {dataset} is {best_model} with score {best_model_score}")

            batch_size = best_model.split("_")[0][2:]
            learning_rate = best_model.split("_")[1][2:]
            embed_size = best_model.split("_")[2][2:]
            
            if dataset not in best_model_metrics:
                best_model_metrics[dataset] = {}
            best_model_metrics[dataset][model] = best_metrics
            
            if model != "cnn-rnn":
                num_layers = best_model.split("_")[3][2:]
            
            if model == "cnn-rnn":
                cmd = f"python eval.py --batch_size={batch_size} --learning_rate={learning_rate} --embed_size={embed_size} \
                --model_arch={model} --dataset={dataset} \
                --checkpoint_dir=./checkpoints/{model}/{dataset}/{best_model}/checkpoint_epoch_{num_epoch_eval}.pth.tar"
            else:
                cmd = f"python eval.py --batch_size={batch_size} --learning_rate={learning_rate} --embed_size={embed_size} \
                    --num_layers={num_layers} --model_arch={model} --dataset={dataset} \
                    --checkpoint_dir=./checkpoints/{model}/{dataset}/{best_model}/checkpoint_epoch_{num_epoch_eval}.pth.tar"
            subprocess.call(cmd, shell=True)
                    
    create_metric_plots(best_model_metrics)
    select_n_samples(n, model, dataset)