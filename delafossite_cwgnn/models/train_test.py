import os
import torch
import torch.nn.functional as F
import csv
from sklearn.metrics import r2_score, accuracy_score

def evaluate(model, dataloader, device, concepts_are_continuous=True):
    """
    Evaluate model on dataset.
    Returns:
        cls_metric: classification accuracy
        concept_metrics: list of R² per concept (if continuous)
    """
    model.eval()
    all_labels, all_preds = [], []
    all_concept_labels, all_concept_preds = [], []

    with torch.no_grad():
        for g, labels, concept_labels in dataloader:
            g, labels, concept_labels = g.to(device), labels.to(device), concept_labels.to(device)

            if hasattr(model, "change_mode"):
                model.change_mode(-1)  # whitening mode

            _, logits, concept_logits = model(g, g.ndata['feat'].to(device))
            preds = logits.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_concept_labels.append(concept_labels.cpu())
            all_concept_preds.append(concept_logits.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_concept_labels = torch.cat(all_concept_labels)
    all_concept_preds = torch.cat(all_concept_preds)

    cls_metric = accuracy_score(all_labels.numpy(), all_preds.numpy())

    concept_metrics = []
    if concepts_are_continuous:
        for i in range(all_concept_labels.shape[1]):
            r2 = r2_score(all_concept_labels[:, i].numpy(), all_concept_preds[:, i].numpy())
            concept_metrics.append(r2)
    else:
        # multi-label binary
        concept_pred_binary = (torch.sigmoid(all_concept_preds) >= 0.5).int()
        for i in range(all_concept_labels.shape[1]):
            concept_acc = (concept_pred_binary[:, i] == all_concept_labels[:, i].int()).float().mean().item()
            concept_metrics.append(concept_acc)

    return cls_metric, concept_metrics


def train_model(model, train_loader, val_loader, test_loader, device,
                optimizer, scheduler=None, num_epochs=50, patience=5,
                save_path="checkpoints/best_model.pth",
                concepts_are_continuous=True,
                early_stopping=True,
                concept_loss_weights=None,
                cls_loss_weight=1.0):
    """
    Training loop with per-concept loss and logging.
    """
    model = model.to(device)
    best_val_acc = 0.0
    epochs_no_improve = 0
    metrics = []

    num_concepts = model.concept_head.out_features if hasattr(model, "concept_head") else 0
    uses_cw = hasattr(model, "change_mode") and hasattr(model, "update_rotation_matrix")

    if concept_loss_weights is None:
        concept_loss_weights = [1.0] * num_concepts

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_cls_loss = 0.0
        total_concept_loss = 0.0
        total_correct = 0
        total_samples_cls = 0

        # Per-concept losses
        concept_losses_epoch = [0.0] * num_concepts

        concept_indices = range(num_concepts) if uses_cw else [None]

        for concept_idx in concept_indices:
            if uses_cw:
                model.change_mode(concept_idx)

            for g, labels, concept_labels in train_loader:
                g, labels, concept_labels = g.to(device), labels.to(device), concept_labels.to(device)

                optimizer.zero_grad()
                _, logits, concept_logits = model(g, g.ndata['feat'].to(device))

                loss = 0.0

                if concept_idx is not None:
                    w = concept_loss_weights[concept_idx]
                    loss_concept = F.mse_loss(concept_logits[:, concept_idx],
                                              concept_labels[:, concept_idx].float()) * w
                    loss += loss_concept
                    concept_losses_epoch[concept_idx] += loss_concept.item() * labels.size(0)

                if (concept_idx is None) or (concept_idx == 0):
                    loss_cls = F.cross_entropy(logits, labels) * cls_loss_weight
                    loss += loss_cls
                    total_cls_loss += loss_cls.item() * labels.size(0)
                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples_cls += labels.size(0)

                loss.backward()
                optimizer.step()

            if uses_cw:
                model.update_rotation_matrix()

        if uses_cw:
            model.change_mode(-1)

        # Compute averages
        train_cls_loss = total_cls_loss / max(1, total_samples_cls)
        train_concept_losses = [l / max(1, len(train_loader.dataset)) for l in concept_losses_epoch]
        train_acc = total_correct / max(1, total_samples_cls)

        # Eval
        val_cls_acc, val_concept_metrics = evaluate(model, val_loader, device, concepts_are_continuous)
        test_cls_acc, test_concept_metrics = evaluate(model, test_loader, device, concepts_are_continuous)

        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_cls_acc)

        print(f"Epoch {epoch:02d} | Train CLS Loss: {train_cls_loss:.4f} "
              f"Train Concept Losses: {train_concept_losses} Acc: {train_acc:.4f}")
        print(f"          Val CLS Acc: {val_cls_acc:.4f} Val Concept Metrics: {val_concept_metrics}")
        print(f"          Test CLS Acc: {test_cls_acc:.4f} Test Concept Metrics: {test_concept_metrics}")
        print(f"          Concept Loss Weights: {concept_loss_weights}")
        print(f"          LR: {optimizer.param_groups[0]['lr']:.6f}")

        metrics.append({
            'epoch': epoch,
            'train_cls_loss': train_cls_loss,
            'train_concept_losses': train_concept_losses,
            'train_acc': train_acc,
            'val_cls_acc': val_cls_acc,
            'val_concept_metrics': val_concept_metrics,
            'test_cls_acc': test_cls_acc,
            'test_concept_metrics': test_concept_metrics,
        })

        # Early stopping
        monitor_value = val_cls_acc
        if monitor_value > best_val_acc:
            best_val_acc = monitor_value
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"Saved best model at epoch {epoch}")
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break

    # Save metrics
    csv_path_metrics = os.path.splitext(save_path)[0] + "_metrics.csv"
    with open(csv_path_metrics, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_cls_loss', 'train_concept_losses', 'train_acc',
                      'val_cls_acc', 'val_concept_metrics', 'test_cls_acc', 'test_concept_metrics']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

    print("Training finished.")
    print(f"Best val classification accuracy (monitor): {best_val_acc:.4f}")
    print(f"Training metrics saved to {csv_path_metrics}")
    return metrics
