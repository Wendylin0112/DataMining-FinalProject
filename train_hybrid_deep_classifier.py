import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


REQUESTED_COMPONENT_ORDER = [
    "vari-grip",
    "lightning-rod-suspension",
    "polymer-insulator-upper-shackle",
    "glass-insulator",
    "yoke-suspension",
]

STATUS_TO_LABEL = {"good": 0, "bad": 1}
LABEL_TO_STATUS = {0: "normal/good", 1: "defective/bad"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a pretrained component classifier plus hybrid per-component defect classifiers."
    )
    parser.add_argument("--train-dir", default="train_dataset")
    parser.add_argument("--test-dir", default="test_dataset")
    parser.add_argument("--template", default="test_submission_template.csv")
    parser.add_argument("--output", default="test_submission_hybrid_deep.csv")
    parser.add_argument("--detailed-output", default="test_predictions_hybrid_deep_detailed.csv")
    parser.add_argument("--model-dir", default="hybrid_deep_models")
    parser.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "mobilenet_v3_large"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--component-epochs", type=int, default=6)
    parser.add_argument("--defect-epochs", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-finetune", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-defect-precision", type=float, default=0.90)
    parser.add_argument(
        "--binary-sampler",
        choices=["label", "component_defect"],
        default="component_defect",
        help="Sampling strategy for binary defect models. component_defect helps rare asset-defect groups.",
    )
    parser.add_argument(
        "--sampler-max-weight-ratio",
        type=float,
        default=12.0,
        help="Cap sampler weights to avoid over-repeating tiny groups. Set 0 to disable capping.",
    )
    parser.add_argument(
        "--defect-pos-weight-multiplier",
        type=float,
        default=1.0,
        help="Multiply BCE positive-class weight. Values above 1.0 make the model more recall-oriented.",
    )
    parser.add_argument("--specialist-min-good", type=int, default=100)
    parser.add_argument("--specialist-min-bad", type=int, default=100)
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print training progress every N batches. Set 0 to disable batch-level logs.",
    )
    return parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)


def format_duration(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def natural_key(path_or_name):
    name = Path(path_or_name).stem
    return (0, int(name)) if name.isdigit() else (1, str(path_or_name))


def list_components(train_dir):
    train_dir = Path(train_dir)
    present = {p.name for p in train_dir.iterdir() if p.is_dir()}
    ordered = [name for name in REQUESTED_COMPONENT_ORDER if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def build_train_frame(train_dir):
    train_dir = Path(train_dir)
    rows = []
    components = list_components(train_dir)
    for component in components:
        for status, label in STATUS_TO_LABEL.items():
            status_dir = train_dir / component / status
            if not status_dir.exists():
                continue
            for image_path in sorted(status_dir.glob("*.jpg"), key=natural_key):
                rows.append(
                    {
                        "path": str(image_path),
                        "component": component,
                        "defect_label": label,
                        "status": status,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No training images found under {train_dir}")
    return pd.DataFrame(rows), components


def load_template(template_path, test_dir):
    template_path = Path(template_path)
    test_image_dir = Path(test_dir) / "images"
    if template_path.exists():
        template = pd.read_csv(template_path)
        if "filename" not in template.columns:
            raise ValueError(f"{template_path} must contain a filename column.")
    else:
        template = pd.DataFrame({"filename": [p.name for p in sorted(test_image_dir.glob("*.jpg"), key=natural_key)]})
    paths = [test_image_dir / str(filename) for filename in template["filename"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing test image listed in template: {missing[0]}")
    return template, [str(path) for path in paths]


def make_transforms(image_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


class PowerDataset(Dataset):
    def __init__(self, frame, transform, target_col=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        if self.target_col is None:
            return image, index
        label = row[self.target_col]
        return image, torch.tensor(label, dtype=torch.long)


class PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), index


def get_head_module(model, arch):
    if arch.startswith("efficientnet") or arch.startswith("mobilenet"):
        return model.classifier
    if arch.startswith("resnet"):
        return model.fc
    raise ValueError(f"Unsupported arch: {arch}")


def replace_head(model, arch, num_classes, dropout):
    if arch.startswith("efficientnet") or arch.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def create_model(arch, num_classes, pretrained=True, dropout=0.25):
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return replace_head(model, arch, num_classes, dropout)


def set_backbone_trainable(model, arch, trainable):
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in get_head_module(model, arch).parameters():
        parameter.requires_grad = True


def make_optimizer(model, lr, weight_decay):
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def make_sample_weights(frame, target_col, sampler_mode, max_weight_ratio):
    if sampler_mode == "label":
        groups = frame[target_col].astype(str)
    elif sampler_mode == "component_defect":
        if "component" not in frame.columns:
            raise ValueError("component_defect sampler requires a component column.")
        groups = frame["component"].astype(str) + "_" + frame[target_col].astype(str)
    else:
        raise ValueError(f"Unsupported sampler mode: {sampler_mode}")

    counts = groups.value_counts()
    weights_by_group = len(groups) / (len(counts) * counts)
    sample_weights = groups.map(weights_by_group).to_numpy(dtype=np.float64)

    if max_weight_ratio and max_weight_ratio > 0:
        median_weight = float(np.median(sample_weights))
        max_weight = median_weight * max_weight_ratio
        sample_weights = np.minimum(sample_weights, max_weight)

    return sample_weights


def make_loader(
    frame,
    transform,
    target_col,
    batch_size,
    num_workers,
    shuffle=False,
    sampler_mode=None,
    max_weight_ratio=12.0,
):
    dataset = PowerDataset(frame, transform, target_col=target_col)
    sampler = None
    if sampler_mode:
        sample_weights = make_sample_weights(frame, target_col, sampler_mode, max_weight_ratio)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_ce_weights(labels, num_classes, device):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_pos_weight(labels, device, multiplier=1.0):
    labels = np.asarray(labels, dtype=int)
    positives = max(int(labels.sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    return torch.tensor([(negatives / positives) * multiplier], dtype=torch.float32, device=device)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    binary_task,
    model_name,
    epoch_number,
    total_epochs,
    log_every,
):
    model.train()
    total_loss = 0.0
    total_count = 0
    use_amp = device.type == "cuda"
    total_batches = len(loader)
    epoch_start = time.time()
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if binary_task:
                logits = logits.squeeze(1)
                loss = criterion(logits, labels.float())
            else:
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        if log_every > 0 and (batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches):
            avg_loss = total_loss / max(total_count, 1)
            elapsed = format_duration(time.time() - epoch_start)
            print(
                f"    [{model_name}] epoch {epoch_number}/{total_epochs} "
                f"batch {batch_index}/{total_batches} "
                f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} elapsed={elapsed}",
                flush=True,
            )
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_logits(model, paths_or_frame, transform, batch_size, num_workers, device):
    if isinstance(paths_or_frame, pd.DataFrame):
        dataset = PathDataset(paths_or_frame["path"].tolist(), transform)
    else:
        dataset = PathDataset(paths_or_frame, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        chunks.append(logits.detach().cpu())
    return torch.cat(chunks).numpy()


def threshold_for_precision(y_true, scores, target_precision):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.r_[0.0, np.unique(scores), 1.0 + 1e-8]
    best = None
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        positives = int(predictions.sum())
        if positives == 0:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            predictions,
            labels=[1],
            average="binary",
            zero_division=0,
        )
        accuracy = accuracy_score(y_true, predictions)
        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "positives": positives,
        }
        if precision >= target_precision:
            if best is None or (candidate["recall"], candidate["f1"], candidate["accuracy"]) > (
                best["recall"],
                best["f1"],
                best["accuracy"],
            ):
                best = candidate
    if best is not None:
        return best

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        positives = int(predictions.sum())
        if positives == 0:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            predictions,
            labels=[1],
            average="binary",
            zero_division=0,
        )
        accuracy = accuracy_score(y_true, predictions)
        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "positives": positives,
        }
        if best is None or (candidate["precision"], candidate["recall"], candidate["f1"]) > (
            best["precision"],
            best["recall"],
            best["f1"],
        ):
            best = candidate
    if best is None:
        return {"threshold": 1.0 + 1e-8, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "positives": 0}
    return best


def evaluate_component(model, frame, transform, batch_size, num_workers, device, idx_to_component):
    logits = predict_logits(model, frame, transform, batch_size, num_workers, device)
    pred_idx = logits.argmax(axis=1)
    pred_components = np.asarray([idx_to_component[index] for index in pred_idx])
    accuracy = accuracy_score(frame["component"].to_numpy(), pred_components)
    return accuracy, pred_components


def evaluate_binary_model(model, frame, transform, batch_size, num_workers, device, target_precision):
    logits = predict_logits(model, frame, transform, batch_size, num_workers, device).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-logits))
    result = threshold_for_precision(frame["defect_label"].to_numpy(), scores, target_precision)
    predictions = (scores >= result["threshold"]).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["defect_label"].to_numpy(),
        predictions,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": result["threshold"],
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(frame["defect_label"].to_numpy(), predictions)),
    }


def train_model(
    name,
    arch,
    num_classes,
    train_frame,
    valid_frame,
    target_col,
    train_transform,
    eval_transform,
    args,
    device,
    binary_task=False,
):
    model = create_model(arch, num_classes=num_classes, pretrained=not args.no_pretrained)
    model = model.to(device)
    set_backbone_trainable(model, arch, trainable=args.freeze_epochs <= 0)

    if binary_task:
        sampler_mode = args.binary_sampler if train_frame["component"].nunique() > 1 else "label"
    else:
        sampler_mode = "label"

    train_loader = make_loader(
        train_frame,
        train_transform,
        target_col,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        sampler_mode=sampler_mode,
        max_weight_ratio=args.sampler_max_weight_ratio,
    )

    if binary_task:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=compute_pos_weight(train_frame[target_col], device, args.defect_pos_weight_multiplier)
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=compute_ce_weights(train_frame[target_col], num_classes, device))

    optimizer = make_optimizer(model, args.lr_head, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -math.inf
    best_state = None
    best_metrics = {}
    epochs = args.defect_epochs if binary_task else args.component_epochs

    print(
        f"\nTraining {name}: {len(train_frame)} train / {len(valid_frame)} valid "
        f"(sampler={sampler_mode})",
        flush=True,
    )
    for epoch in range(epochs):
        if epoch == args.freeze_epochs:
            set_backbone_trainable(model, arch, trainable=True)
            optimizer = make_optimizer(model, args.lr_finetune, args.weight_decay)

        phase = "head-only warmup" if epoch < args.freeze_epochs else "backbone fine-tuning"
        epoch_number = epoch + 1
        print(
            f"  [{name}] starting epoch {epoch_number}/{epochs} ({phase})",
            flush=True,
        )
        epoch_start = time.time()
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            binary_task,
            name,
            epoch_number,
            epochs,
            args.log_every,
        )
        train_time = format_duration(time.time() - epoch_start)

        if binary_task:
            metrics = evaluate_binary_model(
                model,
                valid_frame,
                eval_transform,
                args.batch_size,
                args.num_workers,
                device,
                args.target_defect_precision,
            )
            score = metrics["recall"] if metrics["precision"] >= args.target_defect_precision else metrics["precision"] - 1.0
            print(
                f"  [{name}] finished epoch {epoch_number:02d}/{epochs} in {train_time} "
                f"loss={loss:.4f} "
                f"acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} threshold={metrics['threshold']:.4f}",
                flush=True,
            )
        else:
            accuracy, _ = evaluate_component(
                model,
                valid_frame,
                eval_transform,
                args.batch_size,
                args.num_workers,
                device,
                {idx: component for component, idx in train_frame.attrs["component_to_idx"].items()},
            )
            metrics = {"accuracy": float(accuracy)}
            score = accuracy
            print(
                f"  [{name}] finished epoch {epoch_number:02d}/{epochs} in {train_time} "
                f"loss={loss:.4f} accuracy={accuracy:.4f}",
                flush=True,
            )

        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"  [{name}] best checkpoint updated at epoch {epoch_number}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def tune_route_thresholds(routes, y_true, scores, target_precision):
    thresholds = {}
    results = {}
    global_result = threshold_for_precision(y_true, scores, target_precision)
    for route in sorted(set(routes)):
        mask = routes == route
        if mask.sum() < 5 or y_true[mask].sum() == 0:
            thresholds[route] = global_result["threshold"]
            results[route] = {"used_fallback": True, **global_result}
            continue
        result = threshold_for_precision(y_true[mask], scores[mask], target_precision)
        thresholds[route] = result["threshold"]
        results[route] = {"used_fallback": False, **result}
    return thresholds, results, global_result


def routed_defect_scores(models_by_route, specialist_components, frame_or_paths, pred_components, transform, args, device):
    all_scores = {}
    for route, model in models_by_route.items():
        logits = predict_logits(model, frame_or_paths, transform, args.batch_size, args.num_workers, device).reshape(-1)
        all_scores[route] = 1.0 / (1.0 + np.exp(-logits))

    routes = np.asarray([component if component in specialist_components else "global" for component in pred_components])
    scores = np.asarray([all_scores[route][index] for index, route in enumerate(routes)], dtype=float)
    return routes, scores


def predict_with_thresholds(routes, scores, thresholds, fallback_threshold):
    return np.asarray(
        [int(score >= thresholds.get(route, fallback_threshold)) for route, score in zip(routes, scores)],
        dtype=int,
    )


def print_dataset_summary(frame, components):
    summary = (
        frame.groupby(["component", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=components, columns=["good", "bad"], fill_value=0)
    )
    print("Dataset summary")
    print(summary)


def save_model_bundle(model_dir, component_model, models_by_route, metadata):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(component_model.state_dict(), model_dir / "component_model.pt")
    for route, model in models_by_route.items():
        torch.save(model.state_dict(), model_dir / f"defect_{route}.pt")
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    joblib.dump(metadata, model_dir / "metadata.joblib")


def main():
    configure_console_output()
    args = parse_args()
    seed_everything(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_transform, eval_transform = make_transforms(args.image_size)
    frame, components = build_train_frame(args.train_dir)
    component_to_idx = {component: index for index, component in enumerate(components)}
    idx_to_component = {index: component for component, index in component_to_idx.items()}
    frame["component_idx"] = frame["component"].map(component_to_idx).astype(int)
    print_dataset_summary(frame, components)

    stratify_key = frame["component"].astype(str) + "_" + frame["defect_label"].astype(str)
    train_idx, valid_idx = train_test_split(
        np.arange(len(frame)),
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=stratify_key,
    )
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    valid_frame = frame.iloc[valid_idx].reset_index(drop=True)
    train_frame.attrs["component_to_idx"] = component_to_idx
    valid_frame.attrs["component_to_idx"] = component_to_idx

    component_model, component_metrics = train_model(
        "component classifier",
        args.arch,
        len(components),
        train_frame,
        valid_frame,
        "component_idx",
        train_transform,
        eval_transform,
        args,
        device,
        binary_task=False,
    )
    component_accuracy, valid_pred_components = evaluate_component(
        component_model,
        valid_frame,
        eval_transform,
        args.batch_size,
        args.num_workers,
        device,
        idx_to_component,
    )

    models_by_route = {}
    global_model, global_metrics = train_model(
        "global defect classifier",
        args.arch,
        1,
        train_frame,
        valid_frame,
        "defect_label",
        train_transform,
        eval_transform,
        args,
        device,
        binary_task=True,
    )
    models_by_route["global"] = global_model

    specialist_components = []
    specialist_metrics = {}
    print("\nSpecialist eligibility")
    for component in components:
        component_train = train_frame[train_frame["component"] == component].reset_index(drop=True)
        component_valid = valid_frame[valid_frame["component"] == component].reset_index(drop=True)
        counts = component_train["defect_label"].value_counts().to_dict()
        good_count = int(counts.get(0, 0))
        bad_count = int(counts.get(1, 0))
        eligible = good_count >= args.specialist_min_good and bad_count >= args.specialist_min_bad
        print(f"  {component}: good={good_count} bad={bad_count} eligible={eligible}")
        if not eligible:
            continue
        component_train.attrs["component_to_idx"] = component_to_idx
        component_valid.attrs["component_to_idx"] = component_to_idx
        specialist_model, metrics = train_model(
            f"{component} defect specialist",
            args.arch,
            1,
            component_train,
            component_valid,
            "defect_label",
            train_transform,
            eval_transform,
            args,
            device,
            binary_task=True,
        )
        models_by_route[component] = specialist_model
        specialist_components.append(component)
        specialist_metrics[component] = metrics

    valid_routes, valid_scores = routed_defect_scores(
        models_by_route,
        set(specialist_components),
        valid_frame,
        valid_pred_components,
        eval_transform,
        args,
        device,
    )
    y_valid = valid_frame["defect_label"].to_numpy(dtype=int)
    route_thresholds, route_results, global_threshold_result = tune_route_thresholds(
        valid_routes,
        y_valid,
        valid_scores,
        args.target_defect_precision,
    )
    valid_pred_defects = predict_with_thresholds(
        valid_routes,
        valid_scores,
        route_thresholds,
        global_threshold_result["threshold"],
    )

    defect_precision, defect_recall, defect_f1, _ = precision_recall_fscore_support(
        y_valid,
        valid_pred_defects,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    defect_accuracy = accuracy_score(y_valid, valid_pred_defects)
    component_wrong = valid_pred_components != valid_frame["component"].to_numpy()
    defect_wrong = valid_pred_defects != y_valid

    print("\nHybrid validation results")
    print(f"  component accuracy: {component_accuracy:.4f}")
    print(f"  defect accuracy:    {defect_accuracy:.4f}")
    print(f"  defect precision:   {defect_precision:.4f}")
    print(f"  defect recall:      {defect_recall:.4f}")
    print(f"  defect f1-score:    {defect_f1:.4f}")
    print("  error overlap:")
    print(f"    component only: {int((component_wrong & ~defect_wrong).sum())}")
    print(f"    defect only:    {int((~component_wrong & defect_wrong).sum())}")
    print(f"    both wrong:     {int((component_wrong & defect_wrong).sum())}")
    print("\nDefect classification report")
    print(
        classification_report(
            y_valid,
            valid_pred_defects,
            labels=[0, 1],
            target_names=[LABEL_TO_STATUS[0], LABEL_TO_STATUS[1]],
            zero_division=0,
        )
    )
    print("Route thresholds")
    for route, result in route_results.items():
        suffix = " fallback" if result["used_fallback"] else ""
        print(
            f"  {route}: threshold={route_thresholds[route]:.4f} "
            f"precision={result['precision']:.4f} recall={result['recall']:.4f}{suffix}"
        )

    route_confusion = {}
    print("Route validation confusion")
    for route in sorted(set(valid_routes)):
        route_mask = valid_routes == route
        tn, fp, fn, tp = confusion_matrix(
            y_valid[route_mask],
            valid_pred_defects[route_mask],
            labels=[0, 1],
        ).ravel()
        route_confusion[route] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        print(f"  {route}: tn={tn} fp={fp} fn={fn} tp={tp}")

    template, test_paths = load_template(args.template, args.test_dir)
    test_component_logits = predict_logits(component_model, test_paths, eval_transform, args.batch_size, args.num_workers, device)
    test_component_idx = test_component_logits.argmax(axis=1)
    test_pred_components = np.asarray([idx_to_component[index] for index in test_component_idx])
    test_routes, test_scores = routed_defect_scores(
        models_by_route,
        set(specialist_components),
        test_paths,
        test_pred_components,
        eval_transform,
        args,
        device,
    )
    test_defects = predict_with_thresholds(
        test_routes,
        test_scores,
        route_thresholds,
        global_threshold_result["threshold"],
    )

    submission = template.copy()
    if "pred_label" not in submission.columns:
        submission["pred_label"] = ""
    submission["pred_label"] = test_defects.astype(int)
    submission.to_csv(args.output, index=False)

    detailed = pd.DataFrame(
        {
            "filename": submission["filename"],
            "predicted_component": test_pred_components,
            "defect_route": test_routes,
            "pred_label": test_defects.astype(int),
            "defect_score": test_scores,
        }
    )
    detailed.to_csv(args.detailed_output, index=False)

    metadata = {
        "arch": args.arch,
        "pretrained": not args.no_pretrained,
        "image_size": args.image_size,
        "training_settings": {
            "target_defect_precision": args.target_defect_precision,
            "binary_sampler": args.binary_sampler,
            "sampler_max_weight_ratio": args.sampler_max_weight_ratio,
            "defect_pos_weight_multiplier": args.defect_pos_weight_multiplier,
            "component_epochs": args.component_epochs,
            "defect_epochs": args.defect_epochs,
            "lr_head": args.lr_head,
            "lr_finetune": args.lr_finetune,
            "weight_decay": args.weight_decay,
        },
        "components": components,
        "component_to_idx": component_to_idx,
        "specialist_components": specialist_components,
        "route_thresholds": route_thresholds,
        "route_results": route_results,
        "route_confusion": route_confusion,
        "validation_metrics": {
            "component_accuracy": float(component_accuracy),
            "defect_accuracy": float(defect_accuracy),
            "defect_precision": float(defect_precision),
            "defect_recall": float(defect_recall),
            "defect_f1": float(defect_f1),
        },
        "component_training_metrics": component_metrics,
        "global_defect_metrics": global_metrics,
        "specialist_metrics": specialist_metrics,
    }
    save_model_bundle(args.model_dir, component_model, models_by_route, metadata)

    print("\nTest prediction counts")
    print(detailed.groupby(["defect_route", "pred_label"]).size().unstack(fill_value=0))
    print(f"Saved submission: {args.output}")
    print(f"Saved detailed predictions: {args.detailed_output}")
    print(f"Saved models: {args.model_dir}")


if __name__ == "__main__":
    main()
