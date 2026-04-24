"""
train.py : Training engine for the self-pruning network (soft pruning).

Run:
    python train.py                      # trains all three λ configs
    python train.py --lambda_val 1e-4    # single run
    python train.py --epochs 30 --lr 3e-3

The script saves a checkpoint per lambda value and writes results to
results.json so evaluate.py can plot without retraining.
"""

import argparse
import json
import time
from pathlib import Path
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_loaders
from model import SelfPruningNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, lambda_val, scaler=None, use_tqdm=False):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = correct = seen = 0

    loop = tqdm(loader, desc="Training", leave=False, position=1, dynamic_ncols=True) if use_tqdm else loader
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.amp.autocast(device_type=DEVICE.type):
                logits = model(imgs)
                loss = ce(logits, labels) + lambda_val * model.sparsity_loss()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = ce(logits, labels) + lambda_val * model.sparsity_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += imgs.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        seen    += imgs.size(0)
    return correct / seen


def run(lambda_val: float, epochs: int, lr: float, batch_size: int,
        save_dir: Path, data_dir: str) -> dict:

    print(f"\n{'='*60}")
    print(f"  λ = {lambda_val:.0e}   |   {epochs} epochs   |   device: {DEVICE}")
    print(f"{'='*60}")

    train_loader, val_loader = get_loaders(data_dir, batch_size)
    model = SelfPruningNet(dropout=0.3).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    history = []
    t0 = time.time()

    epoch_loop = tqdm(range(1, epochs + 1), desc=f"Lambda {lambda_val:.0e}", position=0)

    for epoch in epoch_loop:
        epoch_loop.set_description(f"λ={lambda_val:.0e} | Epoch {epoch}/{epochs}")
        use_bar = (epoch % 5 == 0 or epoch == 1 or epoch == epochs)
    
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, lambda_val, scaler, use_tqdm=use_bar
        )
    
        val_acc = evaluate(model, val_loader)
        sparsity_info = model.global_sparsity()
        avg_gate = sum(layer.avg_gate() for layer in model.prunable_layers()) / len(model.prunable_layers())
        scheduler.step()

        history.append({
            "epoch": epoch,
            "tr_loss": round(tr_loss, 4),
            "tr_acc":  round(tr_acc,  4),
            "val_acc": round(val_acc,  4),
            "sparsity": round(sparsity_info["sparsity"], 4),
        })

        if epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            alive = sparsity_info["total"] - sparsity_info["dead"]

            print(f"  ep {epoch:>3}/{epochs}  |  loss {tr_loss:.4f}  |  "
                  f"val {val_acc*100:.2f}%  |  sparse {sparsity_info['sparsity']*100:.1f}%  |  "
                  f"avg_gate {avg_gate:.4f}  |  "
                  f"alive {alive}/{sparsity_info['total']}  |  pruned {sparsity_info['dead']}  |  "
                  f"{elapsed:.0f}s")

    # collect final gate values for the distribution plot
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.extend(torch.sigmoid(layer.gate_scores).detach().cpu().flatten().tolist())

    final_stats = model.global_sparsity()
    result = {
        "lambda": lambda_val,
        "final_val_acc":  history[-1]["val_acc"],
        "final_sparsity": final_stats["sparsity"],
        "dead_weights":   final_stats["dead"],
        "total_weights":  final_stats["total"],
        "history":        history,
        "gate_values":    all_gates,
    }

    ckpt_path = save_dir / f"ckpt_lambda_{lambda_val:.0e}.pt"
    torch.save({"model": model.state_dict(), "result": result}, ckpt_path)
    print(f"\n  ✓  checkpoint → {ckpt_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Network — Tredence Case Study")
    parser.add_argument("--lambda_val", type=float, default=None,
                        help="Single λ to run. If omitted, runs all three benchmark values.")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--out_dir",    type=str,   default="./runs")
    #args = parser.parse_args()
    args, _ = parser.parse_known_args()

    save_dir = Path(args.out_dir)
    save_dir.mkdir(exist_ok=True)

    lambdas = [args.lambda_val] if args.lambda_val else [5e-5, 2e-4, 1e-3]

    all_results = []
    for lam in lambdas:
        res = run(lam, args.epochs, args.lr, args.batch_size, save_dir, args.data_dir)
        all_results.append(res)

    out_path = save_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Lambda':<12} {'Val Acc':>10} {'Sparsity':>12} {'Dead Weights':>15}")
    print(f"  {'-'*52}")
    for r in all_results:
        print(f"  {r['lambda']:<12.0e} {r['final_val_acc']*100:>9.2f}%"
              f" {r['final_sparsity']*100:>11.1f}%"
              f" {r['dead_weights']:>10}/{r['total_weights']}")

    print(f"\n  Results saved → {out_path}")
    print("  Run  python evaluate.py  to generate plots and the markdown report.")


if __name__ == "__main__":
    main()