def train_model(model, name, epochs=60, lr=3e-4, use_adamw=True, weight_decay=1e-4,
                use_scheduler=True, step_size=20, gamma=0.5):
    model = model.to(device)

    if use_adamw:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    loss_fn = nn.CrossEntropyLoss()

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        correct, total, loss_total = 0, 0, 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_total += loss.item() * y.size(0)

        train_acc = correct / total
        train_loss = loss_total / total
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # ---- Val ----
        model.eval()
        correct, total, loss_total = 0, 0, 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                out = model(x)
                loss = loss_fn(out, y)

                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                loss_total += loss.item() * y.size(0)

                val_preds.extend(pred.cpu().tolist())
                val_labels.extend(y.cpu().tolist())

        val_acc = correct / total
        val_loss = loss_total / total
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        cur_lr = opt.param_groups[0]["lr"]
        print(f"[{name}] Epoch {epoch+1:03d}/{epochs} | "
              f"LR {cur_lr:.2e} | "
              f"Train Acc {train_acc:.4f} Loss {train_loss:.4f} | "
              f"Val Acc {val_acc:.4f} Loss {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.show()

    cm = confusion_matrix(val_labels, val_preds, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=train_ds.classes)
    plt.figure(figsize=(10, 10))
    disp.plot(include_values=False, xticks_rotation="vertical")
    plt.title(f"{name} Val Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return model, best_val_acc


@torch.no_grad()
def eval_test(model, name):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    correct, total, loss_total = 0, 0, 0.0
    preds, labels = [], []

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = loss_fn(out, y)

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_total += loss.item() * y.size(0)

        preds.extend(pred.cpu().tolist())
        labels.extend(y.cpu().tolist())

    acc = correct / total
    avg_loss = loss_total / total
    print(f"[{name}] TEST Acc {acc:.4f} | Loss {avg_loss:.4f}")

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=train_ds.classes)
    plt.figure(figsize=(10, 10))
    disp.plot(include_values=False, xticks_rotation="vertical")
    plt.title(f"{name} Test Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return acc


if __name__ == "__main__":
    # VGG19 + SPP
    model_vgg = VGG19_SPP_MLP(num_classes=num_classes)
    model_vgg, best_val_vgg = train_model(
        model_vgg, "VGG19_SPP",
        epochs=60,
        lr=3e-4,
        use_adamw=True,
        weight_decay=1e-4,
        use_scheduler=True,  # set False if you don't want decay
        step_size=20,
        gamma=0.5
    )
    eval_test(model_vgg, "VGG19_SPP")

    # ResNet18 + SPP
    model_res = ResNet18_SPP_MLP(num_classes=num_classes)
    model_res, best_val_res = train_model(
        model_res, "ResNet18_SPP",
        epochs=60,
        lr=3e-4,
        use_adamw=True,
        weight_decay=1e-4,
        use_scheduler=True,
        step_size=20,
        gamma=0.5
    )
    eval_test(model_res, "ResNet18_SPP")