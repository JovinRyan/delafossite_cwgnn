def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    optimizer,
    scheduler=None,
    num_epochs=50,
    patience=5,
    save_path="checkpoints/best_model.pth",
    concepts_are_continuous=True,
    early_stopping=True,
    concept_loss_weight=1.0,   # single scalar
    cls_loss_weight=1.0
):
    model = model.to(device)
    best_val_acc = 0.0
    epochs_no_improve = 0
    metrics = []

    uses_cw = hasattr(model, "change_mode") and hasattr(model, "update_rotation_matrix")
    num_concepts = model.concept_head.out_features if hasattr(model, "concept_head") else 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_cls_loss = 0.0
        total_concept_loss = 0.0
        total_correct = 0
        total_samples_cls = 0

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
                    loss_concept = F.mse_loss(concept_logits[:, concept_idx],
                                              concept_labels[:, concept_idx].float())
                    loss += loss_concept * concept_loss_weight
                    total_concept_loss += loss_concept.item() * labels.size(0)

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
        train_concept_loss = total_concept_loss / max(1, len(train_loader.dataset))
        train_acc = total_correct / max(1, total_samples_cls)

        # Evaluate
        val_cls_acc, val_concept_metrics = evaluate(model, val_loader, device, concepts_are_continuous)
        test_cls_acc, test_concept_metrics = evaluate(model, test_loader, device, concepts_are_continuous)

        if scheduler is not None:
            scheduler.step(val_cls_acc)

        print(f"Epoch {epoch:02d} | Train CLS Loss: {train_cls_loss:.4f} "
              f"CONCEPT Loss: {train_concept_loss:.4f} Acc: {train_acc:.4f}")
        print(f"          Val CLS Acc: {val_cls_acc:.4f} CONCEPT Metric: {val_concept_metrics}")
        print(f"          Test CLS Acc: {test_cls_acc:.4f} CONCEPT Metric: {test_concept_metrics}")
        print(f"          Concept Loss Weight: {concept_loss_weight}")
        print(f"          LR: {optimizer.param_groups[0]['lr']:.6f}")

        metrics.append({
            'epoch': epoch,
            'train_cls_loss': train_cls_loss,
            'train_concept_loss': train_concept_loss,
            'train_acc': train_acc,
            'val_cls_acc': val_cls_acc,
            'val_concept_metrics': val_concept_metrics,
            'test_cls_acc': test_cls_acc,
            'test_concept_metrics': test_concept_metrics,
        })

        # Early stopping
        if val_cls_acc > best_val_acc:
            best_val_acc = val_cls_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"Saved best model at epoch {epoch}")
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break

    print("Training finished.")
    print(f"Best val classification accuracy: {best_val_acc:.4f}")
    return metrics
