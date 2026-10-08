import time
import torch
import torch.nn as nn
from Dataset import get_sudoku_dataloaders
from Model import SudokuTransformer


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, accum_steps=1):
    """
    Trains the SudokuTransformer for one epoch using masked CrossEntropyLoss,
    Intel XPU Automatic Mixed Precision (AMP), and gradient accumulation.
    """
    model.train()

    running_loss = 0.0
    correct_cells = 0
    total_cells = 0
    exact_board_matches = 0
    total_boards = 0

    start_time = time.time()
    total_batches = len(train_loader)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (puzzles, targets) in enumerate(train_loader):
        puzzles = puzzles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. Mask for unfilled/empty cells (where input puzzle is 0)
        empty_mask = (puzzles == 0)

        # 2. Intel XPU Mixed Precision Forward Pass
        with torch.amp.autocast("xpu", dtype=torch.bfloat16):
            logits = model(puzzles)  # Shape: (B, 9, 9, 9)

            # Flatten tensors for CrossEntropyLoss
            flat_logits = logits.view(-1, 9)
            flat_targets = targets.view(-1)
            flat_mask = empty_mask.view(-1)

            # Compute loss ONLY on empty/missing cells
            loss = criterion(flat_logits[flat_mask], flat_targets[flat_mask])
            loss = loss / accum_steps

        # 3. Scaled Backward Pass
        scaler.scale(loss).backward()

        # 4. Gradient Accumulation & Optimizer Step
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == total_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Accumulate unscaled loss for logging
        running_loss += loss.item() * accum_steps

        # 5. Fast Metrics Calculation
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)  # Shape: (B, 9, 9)

            # Accuracy on missing cells only
            correct_cells += ((preds == targets) & empty_mask).sum().item()
            total_cells += empty_mask.sum().item()

            # Exact full-board solve accuracy (all 81 cells correct)
            board_correct = (preds == targets).view(puzzles.size(0), -1).all(dim=1)
            exact_board_matches += board_correct.sum().item()
            total_boards += puzzles.size(0)

        # Print progress every 200 batches
        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == total_batches:
            avg_loss = running_loss / (batch_idx + 1)
            cell_acc = (100.0 * correct_cells / total_cells) if total_cells > 0 else 0.0
            board_acc = 100.0 * exact_board_matches / total_boards

            print(
                f"Epoch [{epoch:02d}] | Batch [{batch_idx + 1:04d}/{total_batches}] | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Cell Acc (Empty Only): {cell_acc:.2f}% | "
                f"Exact Board Acc: {board_acc:.2f}%"
            )

    elapsed = time.time() - start_time
    epoch_loss = running_loss / total_batches
    epoch_cell_acc = (100.0 * correct_cells / total_cells) if total_cells > 0 else 0.0
    epoch_board_acc = 100.0 * exact_board_matches / total_boards

    print(
        f"--> Epoch [{epoch:02d}] Finished in {elapsed:.1f}s | "
        f"Avg Loss: {epoch_loss:.4f} | "
        f"Cell Acc: {epoch_cell_acc:.2f}% | "
        f"Board Acc: {epoch_board_acc:.2f}%\n"
    )

    return epoch_loss, epoch_cell_acc, epoch_board_acc


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_cells = 0
    total_cells = 0
    exact_board_matches = 0
    total_boards = 0

    for puzzles, targets in val_loader:
        puzzles = puzzles.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(puzzles)
        loss = criterion(logits.view(-1, 9), targets.view(-1))

        running_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)

        correct_cells += (preds == targets).sum().item()
        total_cells += targets.numel()

        board_correct = (preds == targets).view(puzzles.size(0), -1).all(dim=1)
        exact_board_matches += board_correct.sum().item()
        total_boards += puzzles.size(0)

    val_loss = running_loss / len(val_loader)
    val_cell_acc = 100.0 * correct_cells / total_cells
    val_board_acc = 100.0 * exact_board_matches / total_boards

    print(
        f"=== VALIDATION === | Loss: {val_loss:.4f} | "
        f"Cell Acc: {val_cell_acc:.2f}% | "
        f"Exact Board Acc: {val_board_acc:.2f}%\n"
    )

    return val_loss, val_cell_acc, val_board_acc


@torch.no_grad()
def solve_sudoku(model, puzzle_np, device):
    """
    Inference helper: Takes a 9x9 NumPy array with values 0-9 and returns a solved 9x9 array (1-9).
    """
    model.eval()
    puzzle_tensor = torch.tensor(puzzle_np, dtype=torch.long).unsqueeze(0).to(device)

    logits = model(puzzle_tensor)  # (1, 9, 9, 9)
    preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return preds + 1


def main():
    # Hardware device setup (Intel XPU, NVIDIA CUDA, or CPU fallback)
    device = torch.device("xpu")

    # Hyperparameters
    num_epochs = 50
    batch_size = 700
    learning_rate = 5e-4

    # DataLoaders
    train_loader, val_loader, test_loader = get_sudoku_dataloaders(
        h5_path="dataset/dataset.h5",
        batch_size=batch_size,
        num_workers=4
    )

    # Model, Optimizer, Scheduler, Criterion
    model = SudokuTransformer(d_model=256, nhead=8, num_layers=6).to(device)

    # Compile model for Intel XPU acceleration
    try:
        # PyTorch 2.1+ natively supports inductor/openxla backends for XPU
        model = torch.compile(model, backend="inductor")
        print("--> Model successfully compiled with torch.compile!")
    except Exception as e:
        print(f"--> Warning: torch.compile failed on XPU ({e}). Continuing with eager mode.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("xpu")

    best_val_acc = 0.0

    print("Starting Training...\n" + "=" * 50)

    for epoch in range(1, num_epochs + 1):
        # 1. Train step
        train_loss, train_cell_acc, train_board_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )

        # 2. Validation step
        val_loss, val_cell_acc, val_board_acc = evaluate(
            model, val_loader, criterion, device
        )

        # 3. Learning Rate Scheduler Step
        scheduler.step()

        # 4. Save best checkpoint based on full-board solve accuracy
        if val_board_acc > best_val_acc:
            best_val_acc = val_board_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_board_acc": val_board_acc,
                },
                "best_sudoku_transformer.pth",
            )
            print(f"--> Saved new best checkpoint! (Val Exact Board Acc: {val_board_acc:.2f}%)\n")


if __name__ == "__main__":
    main()