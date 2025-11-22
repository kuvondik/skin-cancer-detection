# skin_cancer_project.py
# A complete training + evaluation + explainability pipeline for EARLY DIAGNOSIS OF SKIN CANCER (HAM10000)

import os
import csv
import json
import time
import math
import copy
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------
# CONFIG
# --------------------
@dataclass
class Config:
    data_root: str = "./data/HAM10000_custom"  # folder structure: data_root/class_name/*.jpg
    outputs: str = "./outputs"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    patience: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    dropout: float = 0.3
    seed: int = 42
    model_name: str = "efficientnet_b0"
    scheduler: str = "onecycle"  # ['onecycle', 'none']
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    use_class_weights: bool = False
    save_best_model: bool = True
    gradcam_num_samples: int = 6   # how many test images to visualize with Grad-CAM
    lime_num_samples: int = 5      # set >0 only if lime is installed; e.g., 5

CFG = Config()

os.makedirs(CFG.outputs, exist_ok=True)

# Seeds
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

# --------------------
# DATA
# --------------------
CLASS_NAMES = ["AKIEC","BCC","BKL","DF","MEL","NV","VASC"]  # standard 7 classes

def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),    # small rotation only
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def split_indices(n: int, train_p: float, val_p: float, test_p: float):
    idxs = list(range(n))
    random.shuffle(idxs)
    n_train = int(n * train_p)
    n_val = int(n * val_p)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train+n_val]
    test_idx = idxs[n_train+n_val:]
    return train_idx, val_idx, test_idx

class SubsetFolder(torch.utils.data.Dataset):
    def __init__(self, folder_ds: ImageFolder, indices: List[int]):
        self.ds = folder_ds
        self.indices = indices
        self.classes = folder_ds.classes
        self.class_to_idx = folder_ds.class_to_idx

    def __len__(self): 
        return len(self.indices)
    
    def __getitem__(self, i):
        x, y = self.ds[self.indices[i]]
        return x, y

def build_dataloaders(cfg: Config):
    train_tf, eval_tf = get_transforms(cfg.img_size)

    # Base dataset for splitting
    base_ds_for_split = ImageFolder(
        cfg.data_root, 
        transform=transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)), 
            transforms.ToTensor()
        ])
    )
    full_ds_train = ImageFolder(cfg.data_root, transform=train_tf)
    full_ds_eval  = ImageFolder(cfg.data_root, transform=eval_tf)

    n = len(base_ds_for_split)
    train_idx, val_idx, test_idx = split_indices(n, cfg.train_split, cfg.val_split, cfg.test_split)

    ds_train = SubsetFolder(full_ds_train, train_idx)
    ds_val   = SubsetFolder(full_ds_eval,  val_idx)
    ds_test  = SubsetFolder(full_ds_eval,  test_idx)

    # Class weights based on train set distribution
    class_counts = np.zeros(len(base_ds_for_split.classes), dtype=np.int64)
    for i in train_idx:
        _, y = base_ds_for_split[i]
        class_counts[y] += 1

    class_weights = None
    if cfg.use_class_weights:
        # inverse frequency
        class_weights = 1.0 / np.clip(class_counts, 1, None)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Train loader
    if cfg.use_class_weights and class_weights is not None:
        sample_weights = []
        for i in train_idx:
            _, y = base_ds_for_split[i]
            sample_weights.append(class_weights[y].item())
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            ds_train,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=False
        )

    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, class_weights, full_ds_eval.classes

# --------------------
# MODEL
# --------------------
class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3, label_smoothing: float = 0.0):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, num_classes)
        )
        self.label_smoothing = label_smoothing

    def forward(self, x):
        return self.backbone(x)

# --------------------
# TRAINING UTILS
# --------------------
def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, device, criterion, scheduler=None):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * xb.size(0)
        running_acc  += accuracy_from_logits(out, yb) * xb.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def eval_one_epoch(model, loader, device, criterion):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    all_logits, all_targets = [], []
    for xb, yb in tqdm(loader, desc="Eval", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        running_loss += loss.item() * xb.size(0)
        running_acc  += accuracy_from_logits(out, yb) * xb.size(0)
        all_logits.append(out.cpu())
        all_targets.append(yb.cpu())
    n = len(loader.dataset)
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    return running_loss / n, running_acc / n, logits, targets

def plot_training_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Loss
    plt.figure(figsize=(7,5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_validation_loss.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(7,5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"],   label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Training/Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_validation_accuracy.png"))
    plt.close()

def save_params_table(cfg: Config, outdir: str, best_epoch: int):
    params = {
        "Backbone": ["EfficientNet-B0"],
        "Input size": [f"{cfg.img_size}x{cfg.img_size}"],
        "Optimizer": ["AdamW"],
        "Learning rate": [cfg.lr],
        "Weight decay": [cfg.weight_decay],
        "Batch size": [cfg.batch_size],
        "Epochs": [cfg.epochs],
        "Scheduler": ["OneCycleLR" if cfg.scheduler == "onecycle" else "None"],
        "Dropout": [cfg.dropout],
        "Label smoothing": [cfg.label_smoothing],
        "Class weights": ["Yes (inverse freq)" if cfg.use_class_weights else "No"],
        "Augmentations": ["Flip, resize, normalize"],
        "Early stopping patience": [cfg.patience],
        "Best epoch": [best_epoch],
    }
    df = pd.DataFrame(params)
    df.to_csv(os.path.join(outdir, "parameter_selection.csv"), index=False)

# --------------------
# EVALUATION (REPORTS, CONFUSION, ROC, t-SNE)
# --------------------
@torch.no_grad()
def full_evaluation(logits: torch.Tensor, targets: torch.Tensor, class_names: List[str], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    y_true = targets.numpy()
    y_pred = logits.argmax(1).numpy()

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(outdir, "results_table.csv"))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(cm_sum, 1, None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()

    # ROC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    y_score = F.softmax(logits, dim=1).numpy()
    plt.figure(figsize=(7,5))
    aucs = {}
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            auc_val = roc_auc_score(y_true_bin[:, i], y_score[:, i])
            aucs[name] = auc_val
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        except ValueError:
            # class not present in test
            pass
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (OvR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves.png"))
    plt.close()

    # t-SNE on logits (proxy features)
    try:
        tsne = TSNE(
            n_components=2, random_state=42,
            init="random", learning_rate="auto", perplexity=30
        )
        Z = tsne.fit_transform(logits.numpy())
        plt.figure(figsize=(6,6))
        for i, name in enumerate(class_names):
            mask = y_true == i
            plt.scatter(Z[mask,0], Z[mask,1], s=10, alpha=0.7, label=name)
        plt.title("2D Embedding Visualization (t-SNE)")
        plt.legend(markerscale=1.5, bbox_to_anchor=(1.05,1), loc="upper left")
        plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "embedding_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"[WARN] t-SNE failed: {e}")

# --------------------
# GRAD-CAM
# --------------------
class GradCAM:
    """
    Minimal Grad-CAM for EfficientNet last conv stage.
    """
    def __init__(self, model, target_layer_name="features.6.0.block.0"):
        self.model = model.eval()
        self.target_layer = self._get_target_layer(target_layer_name)
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _get_target_layer(self, name: str):
        # Fallback to last features layer if path changes between versions
        m = self.model.backbone.features[-1]
        return m

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def _register_hooks(self):
        self.hook_handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def remove_hooks(self):
        for h in self.hook_handles: 
            h.remove()

    @torch.no_grad()
    def _normalize_cam(self, cam):
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        """
        input_tensor: shape [1,3,H,W] already on correct device
        """
        self.model.zero_grad(set_to_none=True)
        out = self.model(input_tensor)  # [1,C]
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        score = out[:, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Grad-CAM weights
        grads = self.gradients  # [B, C, H, W]
        acts  = self.activations  # [B, C, H, W]
        weights = grads.mean(dim=(2,3), keepdim=True) # [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True) # [B,1,H,W]
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], 
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = self._normalize_cam(cam)
        return cam, target_class

def denorm_to_numpy(t):
    # inverse of ImageNet normalization
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
    x = t.cpu() * std + mean
    x = (x.clamp(0,1) * 255).byte().permute(1,2,0).numpy()
    return x

def overlay_cam(image_np, cam, alpha=0.35):
    heat = plt.cm.jet(cam)[..., :3]  # [H,W,3]
    heat = (heat*255).astype(np.uint8)
    overlay = (alpha*heat + (1-alpha)*image_np).astype(np.uint8)
    return overlay

def gradcam_gallery(model, loader, class_names, outdir, num_images=6, device=None):
    os.makedirs(outdir, exist_ok=True)
    if device is None:
        device = next(model.parameters()).device
    gradcam = GradCAM(model)
    picked = 0
    for xb, yb in loader:
        for i in range(xb.size(0)):
            if picked >= num_images:
                break
            img = xb[i:i+1].to(device)
            cam, pred = gradcam.generate(img)
            img_np = denorm_to_numpy(xb[i])
            overlay = overlay_cam(img_np, cam)
            plt.figure(figsize=(10,4))
            plt.subplot(1,3,1); plt.imshow(img_np); plt.axis("off"); plt.title("Image")
            plt.subplot(1,3,2); plt.imshow(cam, cmap="jet"); plt.axis("off"); plt.title("Grad-CAM")
            plt.subplot(1,3,3); plt.imshow(overlay); plt.axis("off"); plt.title(f"Pred: {class_names[pred]}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"gradcam_{picked}.png"))
            plt.close()
            picked += 1
        if picked >= num_images:
            break
    gradcam.remove_hooks()

# --------------------
# LIME (optional)
# --------------------
def lime_explain_samples(model, loader, class_names, outdir, num_images=3, device=None):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except Exception as e:
        print("[INFO] LIME not installed; skipping LIME visualizations.")
        return
    os.makedirs(outdir, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    def predict_fn(np_imgs):
        # np_imgs: (N, H, W, 3) in [0,255]
        t_list = []
        for x in np_imgs:
            t = transforms.ToTensor()(Image.fromarray(x.astype(np.uint8)))
            t = transforms.Normalize(
                mean=[0.485,0.456,0.406], 
                std=[0.229,0.224,0.225]
            )(t)
            t_list.append(t)
        batch = torch.stack(t_list).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    picked = 0
    for xb, yb in loader:
        for i in range(xb.size(0)):
            if picked >= num_images:
                break
            img_np = denorm_to_numpy(xb[i])  # [H,W,3], uint8
            explanation = explainer.explain_instance(
                img_np,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                label=top_label, positive_only=True, 
                hide_rest=False, num_features=8, min_weight=0.0
            )
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1); plt.imshow(img_np); plt.axis("off"); plt.title("Image")
            plt.subplot(1,2,2); plt.imshow(mark_boundaries(temp/255.0, mask)); plt.axis("off"); plt.title(f"LIME Top: {class_names[top_label]}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"lime_{picked}.png"))
            plt.close()
            picked += 1
        if picked >= num_images:
            break

# --------------------
# MAIN
# --------------------
def main(cfg: Config):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    train_loader, val_loader, test_loader, class_weights, class_names = build_dataloaders(cfg)

    model = EfficientNetB0Classifier(
        num_classes=len(class_names),
        dropout=cfg.dropout,
        label_smoothing=cfg.label_smoothing
    ).to(device)

    # Loss
    if class_weights is not None and cfg.use_class_weights:
        class_weights_t = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t,
            label_smoothing=cfg.label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    if cfg.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, steps_per_epoch=steps_per_epoch, epochs=cfg.epochs
        )
    else:
        scheduler = None

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = cfg.patience
    wait = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, criterion, scheduler)
        val_loss, val_acc, _, _ = eval_one_epoch(model, val_loader, device, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"  train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # Save curves + params table
    plot_training_curves(history, cfg.outputs)
    save_params_table(cfg, cfg.outputs, best_epoch)

    # Restore best
    if best_state is not None and cfg.save_best_model:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(cfg.outputs, "best_model.pth"))

    # Final test evaluation
    test_loss, test_acc, test_logits, test_targets = eval_one_epoch(model, test_loader, device, criterion)
    print(f"\nTest loss {test_loss:.4f} acc {test_acc:.4f}")
    full_evaluation(test_logits, test_targets, class_names, cfg.outputs)

    # Grad-CAM visualizations
    gradcam_gallery(model, test_loader, class_names, cfg.outputs, num_images=cfg.gradcam_num_samples, device=device)

    # LIME (optional)
    if cfg.lime_num_samples > 0:
        lime_explain_samples(model, test_loader, class_names, cfg.outputs, num_images=cfg.lime_num_samples, device=device)

    # Dump a short JSON summary
    summary = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "best_epoch": best_epoch,
        "class_names": class_names
    }
    with open(os.path.join(cfg.outputs, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone. Outputs saved in:", cfg.outputs)

if __name__ == "__main__":
    main(CFG)