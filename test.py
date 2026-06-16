'''
结合5折最优模型与动态相对阈值，进行投票判定，
加入基于归一化熵的不确定性估计，按百分比过滤高风险样本，并保存详细记录。
- 评估指标：AUC, AUPR, Accuracy, Sensitivity, Specificity
- 置信区间计算：DeLong法(AUC), Bootstrap法(AUPR), Wilson法(Acc, Sens, Spec)
'''

import clipmodel.clip as clip
import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score, average_precision_score
)



class MEGC(Dataset):
    def __init__(self, video_dirs, labels, transform=None):
        self.video_dirs = video_dirs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        folder = self.video_dirs[idx]
        label = self.labels[idx]
        try:
            image = Image.open(folder).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法读取图片: {folder}, 错误: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.Resampling.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])



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
        return self.fc(x)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.visual(x)
        y = self.adapter(image_features)
        ratio = 0.6
        image_features = ratio * y + (1 - ratio) * image_features
        x = self.fc1(image_features)
        outputs = self.fcs(x)
        return outputs


def wilson_ci(count, nobs, z=1.95996):
    """使用 Wilson Score Interval 计算单一比例的 95% 置信区间"""
    if nobs == 0:
        return 0.0, 0.0, 0.0
    p = count / nobs
    denominator = 1 + z ** 2 / nobs
    center = (p + z ** 2 / (2 * nobs)) / denominator
    spread = z * math.sqrt(p * (1 - p) / nobs + z ** 2 / (4 * nobs ** 2)) / denominator
    return p, max(0.0, center - spread), min(1.0, center + spread)


def delong_auc_ci(y_true, y_scores, z=1.95996):
    """使用 DeLong's Method 计算 AUC 的 95% 置信区间"""
    auc_val = roc_auc_score(y_true, y_scores)
    pos_preds = y_scores[y_true == 1]
    neg_preds = y_scores[y_true == 0]
    m, n = len(pos_preds), len(neg_preds)

    if m == 0 or n == 0:
        return auc_val, np.nan, np.nan

    v10 = np.array([np.sum(p > neg_preds) + 0.5 * np.sum(p == neg_preds) for p in pos_preds]) / n
    v01 = np.array([np.sum(pos_preds > p) + 0.5 * np.sum(pos_preds == p) for p in neg_preds]) / m

    s1 = np.var(v10, ddof=1) if m > 1 else 0.0
    s0 = np.var(v01, ddof=1) if n > 1 else 0.0

    var_auc = s1 / m + s0 / n
    se = math.sqrt(var_auc)

    return auc_val, max(0.0, auc_val - z * se), min(1.0, auc_val + z * se)


def bootstrap_aupr_ci(y_true, y_scores, n_bootstraps=1000, seed=42):
    """使用 Bootstrapping 法计算 AUPR 的 95% 置信区间"""
    aupr_val = average_precision_score(y_true, y_scores)
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        bootstrapped_scores.append(average_precision_score(y_true[indices], y_scores[indices]))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    if len(sorted_scores) > 0:
        ci_lower = np.percentile(sorted_scores, 2.5)
        ci_upper = np.percentile(sorted_scores, 97.5)
    else:
        ci_lower, ci_upper = np.nan, np.nan

    return aupr_val, ci_lower, ci_upper




def print_evaluation_report_with_95CI(title, labels, preds, scores):
    """计算指标、打印报告并返回"""
    cm = confusion_matrix(labels, preds)
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if labels[0] == 0:
            tn = cm[0][0]
        else:
            tp = cm[0][0]

    acc_val, acc_lower, acc_upper = wilson_ci(tp + tn, tp + tn + fp + fn)
    sens_val, sens_lower, sens_upper = wilson_ci(tp, tp + fn)
    spec_val, spec_lower, spec_upper = wilson_ci(tn, tn + fp)

    if len(np.unique(labels)) > 1:
        auc_val, auc_lower, auc_upper = delong_auc_ci(labels, scores)
        aupr_val, aupr_lower, aupr_upper = bootstrap_aupr_ci(labels, scores)
    else:
        auc_val = aupr_val = auc_lower = auc_upper = aupr_lower = aupr_upper = float('nan')

    print("\n" + "=" * 65)
    print(f"📊 {title}")
    print("=" * 65)
    print(f"1. AUC (DeLong)        : {auc_val:.3f}  (95% CI: {auc_lower:.3f} - {auc_upper:.3f})")
    print(f"2. AUPR (Bootstrap)    : {aupr_val:.3f}  (95% CI: {aupr_lower:.3f} - {aupr_upper:.3f})")
    print("-" * 65)
    print(f"3. Accuracy (Wilson)   : {acc_val:.3f}  (95% CI: {acc_lower:.3f} - {acc_upper:.3f})")
    print(f"4. Sensitivity (Wilson): {sens_val:.3f}  (95% CI: {sens_lower:.3f} - {sens_upper:.3f})")
    print(f"5. Specificity (Wilson): {spec_val:.3f}  (95% CI: {spec_lower:.3f} - {spec_upper:.3f})")
    print("-" * 65)
    print("💡 混淆矩阵 (Confusion Matrix):")
    print(f"                  预测正常(0)   预测肥厚(1)")
    print(f"实际正常(0)  |      {tn:<5}      |      {fp:<5}      |")
    print(f"实际肥厚(1)  |      {fn:<5}      |      {tp:<5}      |")
    print("=" * 65)

    return auc_val


def plot_and_save_roc(labels, scores, roc_auc, save_path):
    """绘制并保存ROC曲线"""
    if not np.isnan(roc_auc):
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Ensemble Model ROC Curve (Filtered)')
        plt.legend(loc="lower right")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 最终ROC曲线已生成并保存至: {os.path.abspath(save_path)}")
        plt.close()


def plot_external_uncertainty_distribution(is_error, uncertainty_scores, dynamic_threshold, save_path):
    """绘制外部测试集的不确定性分布图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    scores_correct = uncertainty_scores[is_error == 0]
    scores_error = uncertainty_scores[is_error == 1]

    plt.hist(scores_correct, bins=30, alpha=0.6, color='blue', label='Correct Prediction (Trust)', density=True)
    plt.hist(scores_error, bins=30, alpha=0.6, color='red', label='Incorrect Prediction (Error)', density=True)
    plt.axvline(dynamic_threshold, color='green', linestyle='--', linewidth=2,
                label=f'Dynamic Threshold: {dynamic_threshold:.4f}')

    plt.title('External Test Set: Uncertainty Score Distribution (Normalized Entropy)')
    plt.xlabel('Uncertainty Score (Normalized Entropy)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📈 外部测试集不确定性分布图已保存至: {save_path}")
    plt.close()



def voting_inference_with_uncertainty(test_excel_path, param_dir, device="cuda:0", target_pass_rate=85):

    print(f"\n[INFO] 启动集成投票推理系统，读取参数目录: {param_dir}")

    df_test = pd.read_excel(test_excel_path)
    X_paths = df_test["file_path"].tolist()
    labels = df_test["label"].values
    test_loader = DataLoader(MEGC(X_paths, labels, transform=transform), batch_size=32, shuffle=False)

    threshold_path = os.path.join(param_dir, "thresholds.json")
    with open(threshold_path, "r") as f:
        thresholds = json.load(f)
    print(f"载入各个Fold的最佳判定阈值: {thresholds}")

    models = []
    print("⏳ 正在独立加载 5 个模型权重，请稍候...")
    for fold in [1, 2, 3, 4, 5]:
        clip_model, _ = clip.load(name="ViT-B/16", device=device, jit=False)
        model = CLIPWithAUClassifier(clip_model, au_num=2, dropout=0.0).float().to(device)
        model_path = os.path.join(param_dir, f"Sub_{fold}_best.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append((f"Fold_{fold}", model))

    all_final_votes = []
    all_true_labels = []
    all_uncertainty_scores = []
    all_avg_probs = []

    print("🚀 开始进行 5 模型集成推理预测...")
    with torch.no_grad():
        for batch in test_loader:
            data_val, labels_val = batch[0].to(device), batch[1].numpy()

            batch_votes = np.zeros((5, len(labels_val)))
            batch_probs = np.zeros((5, len(labels_val)))

            for m_idx, (fold_name, model) in enumerate(models):
                thresh = thresholds[fold_name]
                outputs = model(data_val.float())
                scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                batch_probs[m_idx] = scores
                batch_votes[m_idx] = (scores >= thresh).astype(int)

            sum_votes = np.sum(batch_votes, axis=0)
            final_pred = (sum_votes >= 3).astype(int)

            avg_probs_1 = np.mean(batch_probs, axis=0)
            avg_probs_0 = 1.0 - avg_probs_1

            epsilon = 1e-9
            avg_probs_1_clipped = np.clip(avg_probs_1, epsilon, 1.0 - epsilon)
            avg_probs_0_clipped = np.clip(avg_probs_0, epsilon, 1.0 - epsilon)

            entropy = - (avg_probs_0_clipped * np.log(avg_probs_0_clipped) + avg_probs_1_clipped * np.log(
                avg_probs_1_clipped))
            norm_entropy = entropy / np.log(2)

            all_final_votes.extend(final_pred)
            all_true_labels.extend(labels_val)
            all_uncertainty_scores.extend(norm_entropy)
            all_avg_probs.extend(avg_probs_1)

    all_true_labels = np.array(all_true_labels)
    all_final_votes = np.array(all_final_votes)
    all_avg_probs = np.array(all_avg_probs)
    all_uncertainty_scores = np.array(all_uncertainty_scores)

    print_evaluation_report_with_95CI("初始模型评估报告 (未过滤不确定性样本)", all_true_labels, all_final_votes,
                                      all_avg_probs)

    dynamic_threshold = np.percentile(all_uncertainty_scores, target_pass_rate)
    print(f"\n⚙️ 动态计算的外部测试集熵阈值 ({target_pass_rate}th Percentile): {dynamic_threshold:.4f}")

    confident_mask = all_uncertainty_scores <= dynamic_threshold
    uncertain_samples_count = np.sum(~confident_mask)
    total_samples = len(all_true_labels)

    print(
        f"⚠️ 按 {100 - target_pass_rate}% 的比例，成功拦截高风险不确定性样本: {uncertain_samples_count} 例 (占比 {uncertain_samples_count / total_samples * 100:.1f}%)")

    is_error = (all_final_votes != all_true_labels).astype(int)
    results_df = pd.DataFrame({
        "file_path": X_paths,
        "true_label": all_true_labels,
        "ensemble_pred": all_final_votes,
        "is_error": is_error,
        "uncertainty_score_entropy": all_uncertainty_scores,
        "is_rejected": (~confident_mask).astype(int)
    })

    csv_save_path = os.path.join(param_dir, f"external_samples_analysis_reject{100 - target_pass_rate}.csv")
    results_df.to_csv(csv_save_path, index=False)
    print(f"💾 样本级别详细分析数据(含是否被拦截)已保存至: {csv_save_path}")

    dist_save_path = os.path.join(param_dir, f"External_Uncertainty_Distribution_Filtered{target_pass_rate}.png")
    plot_external_uncertainty_distribution(is_error, all_uncertainty_scores, dynamic_threshold, dist_save_path)

    filtered_true = all_true_labels[confident_mask]
    filtered_preds = all_final_votes[confident_mask]
    filtered_scores = all_avg_probs[confident_mask]

    filtered_auc = print_evaluation_report_with_95CI(
        f"最终模型评估报告 (已过滤最危险的 {100 - target_pass_rate}% 样本)", filtered_true, filtered_preds,
        filtered_scores)

    roc_save_path = os.path.join(param_dir, f"Ensemble_Test_ROC_Curve_Filtered{target_pass_rate}.png")
    plot_and_save_roc(filtered_true, filtered_scores, filtered_auc, roc_save_path)

    return all_final_votes


if __name__ == "__main__":
    # ================= 测试配置与路径指定 =================
    # 1. 测试集 Excel 索引路径（建议存放在项目根目录的 data 文件夹下）
    TEST_EXCEL = "./data/Test_Images_Labels_inner.xlsx"

    # 2. 训练阶段导出的五折最优权重根目录（对应 train.py 中配置的输出目录）
    WEIGHTS_DIR = "./outputs/train_results_vit16/Lr0.001_Rs1_BS16_Alp0.5_Eph100_OlVT-6-6"

    # 自动寻找阈值拦截排名前 % 最不确定的样本，启动集成推理
    voting_inference_with_uncertainty(test_excel_path=TEST_EXCEL, param_dir=WEIGHTS_DIR, target_pass_rate=75)