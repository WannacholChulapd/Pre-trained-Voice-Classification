import torchaudio
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    recall_score,
    precision_score,
)
import numpy as np
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from feature_extract import AudioFeatureExtractor


# ====== CONFIG ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "dataset/train"
num_classes = 2
k_folds = 5

# ====== DATA TRANSFORM ======
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
dataset = datasets.ImageFolder(data_dir, transform=transform)
targets = np.array(dataset.targets)

# ====== CLASS WEIGHTS ======
class_weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")


# ====== CREATE MODEL ======
def create_model(model_name="efficientnet_b0"):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    else:
        raise ValueError("Unknown model name")

    # Replace classifier
    if hasattr(model, "classifier"):
        for i, layer in reversed(list(enumerate(model.classifier))):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                model.classifier[i] = nn.Linear(in_features, num_classes)
                break
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


# ====== TRAIN FUNCTION ======
def train_one_fold(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    fold_idx=0,
    save_dir="results",
):
    os.makedirs(save_dir, exist_ok=True)

    # ===== Compute class weights =====
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    classes = np.unique(all_labels)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"⚖️ Class Weights: {class_weights.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ===== Training Loop =====
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Acc: {acc:.4f}"
        )

    # ===== Validation =====
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())  # prob of class PD

    # ===== Metrics =====
    acc_val = accuracy_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred, average="weighted")
    recall_val = recall_score(y_true, y_pred, average="weighted")
    precision_val = precision_score(y_true, y_pred, average="weighted")

    print(f"[Fold {fold_idx}] Validation Results:")
    print(f"  Accuracy:  {acc_val:.4f}")
    print(f"  Precision: {precision_val:.4f}")
    print(f"  Recall:    {recall_val:.4f}")
    print(f"  F1 Score:  {f1_val:.4f}")

    # ===== Classification Report =====
    report = classification_report(y_true, y_pred, target_names=["HC", "PD"])
    print("\nClassification Report:\n", report)

    # Save report
    report_path = os.path.join(save_dir, f"fold{fold_idx}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"📝 Report saved at: {report_path}")

    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - Fold {fold_idx}")
    cm_path = os.path.join(save_dir, f"fold{fold_idx}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion Matrix saved at: {cm_path}")

    # ===== ROC Curve =====
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - Fold {fold_idx}")
        plt.legend()
        roc_path = os.path.join(save_dir, f"fold{fold_idx}_roc_curve.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        print(f"📈 ROC Curve saved at: {roc_path}")
    except Exception as e:
        print(f"⚠️ ROC Curve generation failed: {e}")

    # ===== Save Model =====
    model_path = os.path.join(save_dir, f"fold{fold_idx}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at: {model_path}")

    return acc_val, f1_val


# ====== K-FOLD TRAIN FOR BOTH MODELS ======
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
models_to_train = ["efficientnet_b0", "mobilenet_v3_small"]
results = []

for model_name in models_to_train:
    print(f"\n🚀 Training model: {model_name}")
    save_dir = os.path.join("results", model_name)
    os.makedirs(save_dir, exist_ok=True)

    fold_acc, fold_f1 = [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n========== {model_name} | Fold {fold+1}/{k_folds} ==========")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # ===== สร้างโมเดลใหม่สำหรับแต่ละ fold =====
        model = create_model(model_name)
        model.to(device)

        # ===== เทรนแต่ละ fold =====
        acc, f1 = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
            device=device,
            fold_idx=fold + 1,
            save_dir=save_dir,
        )

        fold_acc.append(acc)
        fold_f1.append(f1)

    # ===== สรุปผลเฉลี่ยแต่ละโมเดล =====
    results.append(
        {
            "Model": model_name,
            "Acc(mean)": np.mean(fold_acc),
            "F1(mean)": np.mean(fold_f1),
        }
    )

# ===== แสดงผลรวมทุกโมเดล =====
print("\n===== FINAL RESULTS =====")
for res in results:
    print(f"{res['Model']}:  Acc={res['Acc(mean)']:.4f} | F1={res['F1(mean)']:.4f}")


# ====== SAVE FINAL RESULTS ======
df_results = pd.DataFrame(results)
print("\n========== FINAL RESULTS ==========")
print(df_results)
df_results.to_csv("results/final_results.csv", index=False)
