'''
幽门超声图像识别系统—CLIP模型适配器实现
五折验证逻辑、val_auc早停、约登指数最佳阈值、最佳模型保存与汇总调参
'''

import clipmodel.clip as clip
import os
import random
import json
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from datetime import datetime  # [新增] 导入时间模块

pd.set_option("display.max_columns", 50)

def get_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_indexes = tpr - fpr
    best_index = np.argmax(youden_indexes)
    return float(thresholds[best_index])


def plot_roc_curve(y_true, y_score, save_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_loss_curve(train_losses, val_losses, subject, save_path):
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train Total Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Total Loss')
    plt.title(f'Fold/Subject {subject} - Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_acc_curve(train_accs, val_accs, subject, save_path):
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_accs) + 1)
    plt.plot(epochs_range, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy')
    plt.title(f'Fold/Subject {subject} - Train vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 加载数据集索引文件（采用相对路径，可根据实际存放位置修改）
df = pd.read_excel("./data/pylorus_index.xlsx")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class MEGC(Dataset):
    def __init__(self, video_dirs, labels, texts, transform=None):
        self.video_dirs = video_dirs
        self.labels = labels
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        folder = self.video_dirs[idx]
        label = self.labels[idx]
        text = self.texts[idx]
        frame_path = folder
        image = Image.open(frame_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, text


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CLIPWithAUClassifier(nn.Module):
    def __init__(self, clip_model, au_num, dropout=0.5):
        super().__init__()
        self.clip = clip_model
        self.au_num = au_num
        self.dropout = nn.Dropout(dropout)
        self.dtype = clip_model.dtype
        self.adapter = Adapter(clip_model.visual.output_dim, 4).to(clip_model.dtype)
        self.fc1 = nn.Sequential(
            nn.Linear(self.clip.visual.output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fcs = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        image_features = self.clip.visual(x)
        y = self.adapter(image_features)
        ratio = 0.6
        image_features = ratio * y + (1 - ratio) * image_features
        x = self.fc1(image_features)
        outputs = self.fcs(x)
        return outputs, image_features, x


test_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.Resampling.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=Image.Resampling.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])


class BinaryClassificationEvaluator:
    def __init__(self):
        pass

    def evaluate(self, preds, labels, scores=None):
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="binary", zero_division=0)
        recall = recall_score(labels, preds, average="binary", zero_division=0)
        f1 = f1_score(labels, preds, average="binary", zero_division=0)
        auc_score = None
        if scores is not None:
            auc_score = roc_auc_score(labels, scores)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score
        }


def LOSO(df, param_dir, txt_log_path, epochs=100, lr=0.01, batch_size=8, dropout=0.05, weight_decay=0.001, alpha=0.4,
         openlayer=[-1, -1], randomseed=1, ViTName="ViT-B/32", patience=10):
    random.seed(randomseed)
    torch.manual_seed(randomseed)
    np.random.seed(randomseed)
    torch.cuda.manual_seed(randomseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    labels = df["label"].values
    Olayervisual = openlayer[0]
    Olayertext = openlayer[1]

    outputs_listx = []
    labels_list = []
    scores_list = []
    all_subjects_last_val_loss = []
    fold_thresholds = {}

    for group in df.groupby("fold"):
        subject = group[0]
        print(f"\n[{get_time_str()}] [INFO] 开始处理 Subject/Fold: {subject}")

        train_index = np.array(df[df["fold"] != subject].index)
        X_train_paths = df.loc[train_index, "file_path"].tolist()
        y_train = labels[train_index].tolist()
        X_train_texts = df.loc[train_index, "text"].tolist()

        test_index = np.array(df[df["fold"] == subject].index)
        X_test_paths = df.loc[test_index, "file_path"].tolist()
        y_test = labels[test_index].tolist()
        X_test_texts = df.loc[test_index, "text"].tolist()

        megc_dataset_train = MEGC(X_train_paths, y_train, X_train_texts, transform=train_transform)
        dataset_loader_train = torch.utils.data.DataLoader(megc_dataset_train, batch_size=batch_size, shuffle=True,
                                                           drop_last=True)

        megc_dataset_test = MEGC(X_test_paths, y_test, X_test_texts, transform=test_transform)
        dataset_loader_test = torch.utils.data.DataLoader(megc_dataset_test, batch_size=100, shuffle=False)

        clip_model, _ = clip.load(name=ViTName, device=device, jit=False)
        clip_model.float()
        model = CLIPWithAUClassifier(clip_model, au_num=2, dropout=dropout).float().to(device)
        criterion = nn.CrossEntropyLoss()
        evaluator = BinaryClassificationEvaluator()

        for param in model.clip.parameters(): param.requires_grad = False
        for i in range(Olayervisual, 0):
            for param in model.clip.visual.transformer.resblocks[i].parameters(): param.requires_grad = True
        for i in range(Olayertext, 0):
            for param in model.clip.transformer.resblocks[i].parameters(): param.requires_grad = True

        optimizer = optim.SGD([
            {"params": model.adapter.parameters(), "lr": lr},
            {"params": model.fc1.parameters(), "lr": lr},
            {"params": model.fcs.parameters(), "lr": lr},
            {"params": [p for p in model.clip.visual.transformer.resblocks[Olayervisual:].parameters() if
                        p.requires_grad], "lr": lr / 10},
            {"params": [p for p in model.clip.transformer.resblocks[Olayertext:].parameters() if p.requires_grad],
             "lr": lr / 10},
        ], weight_decay=weight_decay, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)

        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []
        val_auc_history = []
        epoch_time_history = []

        best_val_auc = -1.0
        patience_counter = 0
        best_model_path = os.path.join(param_dir, f"Sub_{subject}_best.pt")
        current_fold_best_threshold = 0.5
        best_epoch_record = 0

        for epoch in range(epochs):
            epoch_num = epoch + 1
            model.train()
            epoch_total_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in dataset_loader_train:
                data_batch, labels_batch, texts_batch = batch[0].to(device), batch[1].to(device), batch[2]
                optimizer.zero_grad()
                outputs, image_features, _ = model(data_batch.float())
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                text_tokens = clip.tokenize(texts_batch).to(device)
                text_features = model.clip.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().to(device)
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logits_per_image.T
                ground_truth = torch.arange(len(data_batch)).to(device)
                contrastive_loss = (F.cross_entropy(logits_per_image, ground_truth) +
                                    F.cross_entropy(logits_per_text, ground_truth)) / 2

                loss = criterion(outputs, labels_batch.long())
                total_loss = loss + alpha * contrastive_loss
                total_loss.backward()
                optimizer.step()
                epoch_total_loss += total_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()

            scheduler.step()
            avg_total_loss = epoch_total_loss / len(dataset_loader_train)
            train_acc = train_correct / train_total


            model.eval()
            epoch_val_total_loss = 0.0
            val_correct = 0
            val_total = 0

            all_labels_val_epoch = []
            all_scores_val_epoch = []

            with torch.no_grad():
                for batch_val in dataset_loader_test:
                    data_val, labels_val, texts_val = batch_val[0].to(device), batch_val[1].to(device), batch_val[2]
                    outputs_val, image_features_val, _ = model(data_val.float())
                    image_features_val = image_features_val / image_features_val.norm(dim=1, keepdim=True)

                    text_tokens_val = clip.tokenize(texts_val).to(device)
                    text_features_val = model.clip.encode_text(text_tokens_val)
                    text_features_val = text_features_val / text_features_val.norm(dim=1, keepdim=True)

                    logit_scale_val = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().to(device)
                    logits_per_image_val = logit_scale_val * image_features_val @ text_features_val.T
                    logits_per_text_val = logits_per_image_val.T
                    ground_truth_val = torch.arange(len(data_val)).to(device)
                    contrastive_loss_val = (F.cross_entropy(logits_per_image_val, ground_truth_val) +
                                            F.cross_entropy(logits_per_text_val, ground_truth_val)) / 2

                    loss_val = criterion(outputs_val, labels_val.long())
                    total_loss_val = loss_val + alpha * contrastive_loss_val
                    epoch_val_total_loss += total_loss_val.item()

                    _, predicted_val = torch.max(outputs_val.data, 1)
                    val_total += labels_val.size(0)
                    val_correct += (predicted_val == labels_val).sum().item()

                    scores_batch = F.softmax(outputs_val, dim=1)[:, 1].cpu()
                    all_labels_val_epoch.extend(labels_val.cpu().tolist())
                    all_scores_val_epoch.extend(scores_batch.tolist())

            avg_val_loss = epoch_val_total_loss / len(dataset_loader_test)
            val_acc = val_correct / val_total
            current_val_auc = roc_auc_score(all_labels_val_epoch, all_scores_val_epoch)

            train_loss_history.append(avg_total_loss)
            val_loss_history.append(avg_val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            val_auc_history.append(current_val_auc)


            current_time = get_time_str()
            epoch_time_history.append(current_time)
            print(
                f"[{current_time}]   -> Epoch [{epoch_num:02d}/{epochs}] | Train Loss: {avg_total_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val AUC: {current_val_auc:.4f}")


            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_epoch_record = epoch_num
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                current_fold_best_threshold = find_optimal_threshold(all_labels_val_epoch, all_scores_val_epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"[{get_time_str()}]      [Early Stopping] 在第 {epoch_num} 轮触发早停！最佳 Epoch 为 {best_epoch_record}，AUC: {best_val_auc:.4f}")
                    break

        plot_loss_curve(train_loss_history, val_loss_history, subject,
                        os.path.join(param_dir, f"Sub_{subject}_LossCurve.png"))
        plot_acc_curve(train_acc_history, val_acc_history, subject,
                       os.path.join(param_dir, f"Sub_{subject}_AccCurve.png"))


        with open(txt_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Subject/Fold {subject} ---\n")
            for e_idx in range(len(val_loss_history)):
                f.write(
                    f"[{epoch_time_history[e_idx]}] Epoch {e_idx + 1:02d}: Train_Loss = {train_loss_history[e_idx]:.4f}, Train_Acc = {train_acc_history[e_idx]:.4f} | Val_Loss = {val_loss_history[e_idx]:.4f}, Val_AUC = {val_auc_history[e_idx]:.4f}\n")
            f.write(
                f"[{get_time_str()}] >> Best Model at Epoch {best_epoch_record} with Val_AUC = {best_val_auc:.4f}, Threshold = {current_fold_best_threshold:.4f}\n")


        fold_thresholds[f"Fold_{subject}"] = current_fold_best_threshold


        print(f"[{get_time_str()}] ==> 提取最佳模型 (Epoch {best_epoch_record}) 的结果用于最终评估...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        best_preds_test = []
        best_labels_test = []
        best_scores_test = []
        best_val_loss_accum = 0.0

        with torch.no_grad():
            for batch in dataset_loader_test:
                data_batch_test, labels_batch_test, _ = batch[0].to(device), batch[1].cpu(), batch[2]
                outputs, _, _ = model(data_batch_test.float())

                loss_val = criterion(outputs, labels_batch_test.to(device).long())
                best_val_loss_accum += loss_val.item()

                preds_test = outputs.argmax(dim=1).cpu()
                scores_test = F.softmax(outputs, dim=1)[:, 1].cpu()
                best_preds_test.append(preds_test)
                best_labels_test.append(labels_batch_test)
                best_scores_test.append(scores_test)

        preds_test = torch.cat(best_preds_test, dim=0)
        labels_test = torch.cat(best_labels_test, dim=0)
        scores_test = torch.cat(best_scores_test, dim=0)

        outputs_listx.append(preds_test)
        labels_list.append(labels_test)
        scores_list.append(scores_test)
        all_subjects_last_val_loss.append(best_val_loss_accum / len(dataset_loader_test))


    with open(os.path.join(param_dir, "thresholds.json"), "w") as f:
        json.dump(fold_thresholds, f)


    all_preds = torch.cat(outputs_listx, dim=0).numpy()
    all_labels = torch.cat(labels_list, dim=0).numpy()
    all_scores = torch.cat(scores_list, dim=0).numpy()

    overall_metrics = evaluator.evaluate(all_preds, all_labels, all_scores)
    avg_best_val_loss_overall = sum(all_subjects_last_val_loss) / len(all_subjects_last_val_loss)

    with open(txt_log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n[{get_time_str()}] ====================================\n")
        f.write(f"总体汇总结果 (Best Epochs 组合 Metrics):\n")
        f.write(f"AUC: {overall_metrics['auc']:.4f}\n")
        f.write(f"F1 Score: {overall_metrics['f1']:.4f}\n")
        f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n")
        f.write(f"Avg Best Val Loss: {avg_best_val_loss_overall:.4f}\n")
        f.write(f"====================================\n")

    return all_preds, all_labels, all_scores, overall_metrics, avg_best_val_loss_overall


# ================= 超参数网格搜索配置 =================
# 默认配置会遍历所有组合（当前共计 4 组实验，每组进行五折交叉验证）
# 如果只想快速跑通流程测试，建议将列表中的元素缩减（例如 alp 改为 [0.3]，ol 改为 [-3]）
ParaMeters = []
for rs in [1]:          # 随机种子 (Random Seed)
    for epochs in [100]: # 每个模型训练的总轮数
        for lr in [0.005]: # 基础学习率
            for bs in [16]:  # Batch Size
                for alp in [0.3, 0.5]:   # 对比学习损失权重 alpha
                    for ol in [-6, -3]:  # 解冻的 CLIP Transformer 层数
                        ParaMeters.append([rs, alp, bs, epochs, lr, ol])

print(f"[{get_time_str()}] 总计进行 {len(ParaMeters)} 组超参数实验。")
vitname = "ViT-B/16"

# ================= 实验结果输出根目录 =================
# 训练过程中的模型权重 (.pt)、训练日志 (.txt) 以及可视化曲线图 (.png) 都会自动保存在该目录下
root_dir = "./outputs/train_results_vit16/"
os.makedirs(root_dir, exist_ok=True)
excel_summary_list = []

for PM in ParaMeters:
    Rseed, alp, BatchSize, Ep, lr, ol = PM
    Olayer = [ol, ol]
    param_str = f"Lr{lr}_Rs{Rseed}_BS{BatchSize}_Alp{alp}_Eph{Ep}_OlVT{Olayer[0]}{Olayer[1]}"
    print(f"\n[{get_time_str()}] ▶ 正在执行实验: {param_str}")

    param_dir = os.path.join(root_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)
    txt_log_path = os.path.join(param_dir, "Epoch_ValLoss_Details.txt")
    roc_save_path = os.path.join(param_dir, "Overall_ROC_Curve.png")

    if os.path.exists(txt_log_path):
        os.remove(txt_log_path)
    with open(txt_log_path, 'w', encoding='utf-8') as f:
        f.write(f"[{get_time_str()}] 参数组合: {param_str}\n\n")

    predictions, labels, scores, overall_metrics, avg_best_val_loss = LOSO(
        df, param_dir=param_dir, txt_log_path=txt_log_path,
        epochs=Ep, lr=lr, weight_decay=0.001,
        dropout=0.2, batch_size=BatchSize, alpha=alp,
        openlayer=Olayer, randomseed=Rseed, ViTName=vitname, patience=20
    )

    plot_roc_curve(labels, scores, roc_save_path)

    excel_summary_list.append({
        '参数文件夹名': param_str,
        'Learning_Rate': lr,
        'Batch_Size': BatchSize,
        'Alpha': alp,
        'Unfreeze_Layers': ol,
        'AUC': overall_metrics['auc'],
        'F1_Score': overall_metrics['f1'],
        'Accuracy': overall_metrics['accuracy'],
        'Avg_Best_Val_Loss': avg_best_val_loss
    })

if len(excel_summary_list) > 0:
    df_summary = pd.DataFrame(excel_summary_list)
    df_summary.sort_values(by='AUC', ascending=False, inplace=True)
    excel_save_path = os.path.join(root_dir, "Hyperparameters_Summary.xlsx")
    df_summary.to_excel(excel_save_path, index=False)
    print(f"\n[{get_time_str()}] 全部实验执行完毕！")
    print(f"[{get_time_str()}] 汇总结果（按AUC排序）已保存至: {excel_save_path}")