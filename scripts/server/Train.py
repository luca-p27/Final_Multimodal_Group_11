"""
Train.py — Training and evaluation loops

All functions work with any model that accepts forward(img, geo) -> logits,
matching the unified interface of ContinuousGeoModel, DiscreteEarlyFusionModel,
and DiscreteLateFusionModel.

Public API:
    train_one_epoch(model, loader, optimizer, criterion, device) -> (loss, acc)
    evaluate(model, loader, criterion, device) -> (loss, acc, preds, labels, probs)
    fit(model, train_loader, val_loader, optimizer, scheduler,
        epochs, device, save_path, patience) -> best_val_acc
"""

import copy
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One training pass over loader. Returns (avg_loss, accuracy_%)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc="  Train", leave=False)
    for batch in bar:
        if batch is None:
            continue
        imgs, geos, labels = batch
        imgs, geos, labels = imgs.to(device), geos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, geos)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        bar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluation pass over loader.

    Returns:
        avg_loss   (float)
        accuracy   (float, %)
        all_preds  (list[int])
        all_labels (list[int])
        all_probs  (list[list[float]])
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        if batch is None:
            continue
        imgs, geos, labels = batch
        imgs, geos, labels = imgs.to(device), geos.to(device), labels.to(device)
        outputs = model(imgs, geos)
        loss    = criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (
        total_loss / len(loader),
        100.0 * correct / total,
        all_preds,
        all_labels,
        all_probs,
    )


def fit(model, train_loader, val_loader, optimizer, scheduler,
        epochs: int, device, save_path: str, patience: int = 10):
    """
    Full training loop with early stopping and best-model checkpointing.

    Args:
        model        : nn.Module (or nn.DataParallel wrapper)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        optimizer    : AdamW (or any torch optimizer)
        scheduler    : LR scheduler (called once per epoch)
        epochs       : maximum number of epochs
        device       : torch.device
        save_path    : file path where the best model state dict is saved
        patience     : stop after this many epochs without val improvement

    Returns:
        best_val_acc (float, %)
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc, best_state, no_improve = 0.0, None, 0

    for epoch in range(epochs):
        tr_loss, tr_acc          = train_one_epoch(model, train_loader,
                                                   optimizer, criterion, device)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{epochs}  "
              f"train={tr_acc:.2f}%  val={vl_acc:.2f}%  "
              f"(loss {tr_loss:.4f}/{vl_loss:.4f})")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            print(f"  new best: val={vl_acc:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc