import os
import torch
import torch.nn.functional as F
import csv
from sklearn.metrics import r2_score, accuracy_score

def evaluate(model, dataloader, device, concepts_are_continuous=False,
             cls_metric_fn=None, concept_metric_fn=None):
    """
    Evaluate model on dataset.
    """
    cls_metric_fn = cls_metric_fn or (lambda y_true, y_pred: accuracy_score(y_true, y_pred))
    concept_metric_fn = concept_metric_fn or (lambda y_true, y_pred: r2_score(y_true, y_pred))

    model.eval()
    all_labels, all_preds = [], []
    all_concept_labels, all_concept_preds = [], []

    with torch.no_grad():
        for g, labels, concept_labels in dataloader:
            g = g.to(device)
            feats = g.ndata['feat'].to(device)
            labels = labels.to(device)
            concept_labels = concept_labels.to(device)

            # Ensure whitening mode for eval if available
            if hasattr(model, "change_mode"):
                model.change_mode(-1)

            _, logits, concept_logits = model(g, feats)
            preds = logits.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_concept_labels.append(concept_labels.cpu())
            all_concept_preds.append(concept_logits.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_concept_labels = torch.cat(all_concept_labels)
    all_concept_preds = torch.cat(all_concept_preds)

    cls_metric = cls_metric_fn(all_labels.numpy(), all_preds.numpy())

    if concepts_are_continuous:
        # R² over all concepts at once
        concept_metric = concept_metric_fn(all_concept_labels.numpy(),
                                           all_concept_preds.numpy())
    else:
        # Multi-label concept classification example
        concept_pred_binary = (torch.sigmoid(all_concept_preds) >= 0.5).int()
        concept_metric = (concept_pred_binary == all_concept_labels.int()).float().mean().item()

    return cls_metric, concept_metric


def train_model(model, train_loader, val_loader, test_loader, device,
                optimizer, scheduler=None, num_epochs=50, patience=5,
                save_path="checkpoints/best_model.pth",
                concepts_are_continuous=True,
                early_stopping=True,
                concept_loss_weight=1.0,
                cls_loss_weight=1.0):
    """
    Modular training loop for CWGNN/GNN models.
    - Computes classification loss ONCE per batch (not once per concept).
    - Trains concept loss per concept index for CW rotation.
    """
    model = model.to(device)
    best_val_acc = 0.0
    epochs_no_improve = 0
    metrics = []

    num_concepts = model.concept_head.out_features if hasattr(model, "concept_head") else 0
    uses_cw = hasattr(model, "change_mode") and hasattr(model, "update_rotation_matrix")

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_cls_loss = 0.0
        total_concept_loss = 0.0
        total_correct = 0
        total_samples_cls = 0        # counted once per batch
        total_samples_concept = 0     # counted once per batch for concept loss averaging

        # ---- One pass per concept to train CW rotations ----
        concept_indices = range(num_concepts) if uses_cw else [None]

        for concept_idx in concept_indices:
            if uses_cw:
                model.change_mode(concept_idx)

            for g, labels, concept_labels in train_loader:
                g = g.to(device)
                feats = g.ndata['feat'].to(device)
                labels = labels.to(device)
                concept_labels = concept_labels.to(device)

                optimizer.zero_grad()
                _, logits, concept_logits = model(g, feats)

                # --- concept loss (when applicable) ---
                if concept_idx is not None:  # training per-concept rotation
                    loss_concept = F.mse_loss(
                        concept_logits[:, concept_idx],
                        concept_labels[:, concept_idx].float()
                    ) * concept_loss_weight
                    loss = loss_concept
                    total_concept_loss += loss_concept.item() * labels.size(0)
                    total_samples_concept += labels.size(0)
                else:
                    loss = 0.0

                # --- classification loss: compute ONLY ONCE per batch (concept_idx == first) ---
                if (concept_idx is None) or (concept_idx == 0):
                    loss_cls = F.cross_entropy(logits, labels) * cls_loss_weight
                    loss = loss + loss_cls
                    total_cls_loss += loss_cls.item() * labels.size(0)

                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples_cls += labels.size(0)

                loss.backward()
                optimizer.step()

            # Update rotation after all batches for this concept
            if uses_cw:
                model.update_rotation_matrix()

        # switch to whitening mode for eval
        if uses_cw:
            model.change_mode(-1)

        # Averages
        train_cls_loss = total_cls_loss / max(1, total_samples_cls)
        train_concept_loss = total_concept_loss / max(1, total_samples_concept)
        train_acc = total_correct / max(1, total_samples_cls)

        # Eval
        val_cls_acc, val_concept_metric = evaluate(model, val_loader, device, concepts_are_continuous)
        test_cls_acc, test_concept_metric = evaluate(model, test_loader, device, concepts_are_continuous)

        # Step scheduler (you can also monitor a combo; see commented line)
        if scheduler is not None:
            scheduler.step(val_cls_acc)
            # combined = 0.5 * val_cls_acc + 0.5 * max(val_concept_metric, -1.0)  # if you want to include R²
            # scheduler.step(combined)

        # Nice label for concept metric
        concept_label = "R2" if concepts_are_continuous else "ConceptAcc"
        print(f"Epoch {epoch:02d} | Train CLS Loss: {train_cls_loss:.4f} "
              f"CONCEPT Loss: {train_concept_loss:.4f} Acc: {train_acc:.4f}")
        print(f"          Val CLS Acc: {val_cls_acc:.4f} CONCEPT {concept_label}: {val_concept_metric:.4f}")
        print(f"          Test CLS Acc: {test_cls_acc:.4f} CONCEPT {concept_label}: {test_concept_metric:.4f}")
        print(f"          LR: {optimizer.param_groups[0]['lr']:.6f}")

        metrics.append({
            'epoch': epoch,
            'train_cls_loss': train_cls_loss,
            'train_concept_loss': train_concept_loss,
            'train_acc': train_acc,
            'val_cls_acc': val_cls_acc,
            'val_concept_metric': val_concept_metric,
            'test_cls_acc': test_cls_acc,
            'test_concept_metric': test_concept_metric,
        })

        # Early stopping on val cls acc (or swap to combined criterion)
        monitor_value = val_cls_acc  # or combined
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
        fieldnames = ['epoch', 'train_cls_loss', 'train_concept_loss', 'train_acc',
                      'val_cls_acc', 'val_concept_metric', 'test_cls_acc', 'test_concept_metric']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

    print("Training finished.")
    print(f"Best val classification accuracy (monitor): {best_val_acc:.4f}")
    print(f"Training metrics saved to {csv_path_metrics}")
    return metrics
