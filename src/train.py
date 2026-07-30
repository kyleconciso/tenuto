import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.utils import set_seed, get_device, save_checkpoint
from src.dataset import create_dataloaders
from src.model import build_model

class TenutoLoss(nn.Module):
    r"""
    Multi-objective Loss for Tenuto Expressive AI Engine.
    
    L_total = 100 * L_huber(\Delta t) + 1.0 * L_mse(v) + 10.0 * L_smooth(S_b)
    """
    def __init__(self, w_timing: float = 100.0, w_velocity: float = 1.0, w_smooth: float = 10.0):
        super(TenutoLoss, self).__init__()
        self.w_timing = w_timing
        self.w_velocity = w_velocity
        self.w_smooth = w_smooth
        self.huber = nn.HuberLoss(delta=0.005) # 5ms delta threshold for robust micro-timing loss

    def compute_smoothness_loss(self, tempo_scale):
        r"""
        Second-order difference penalty on beat tempo scale S(b):
        L_smooth = \sum_b |(S_{b+1} - S_b) - (S_b - S_{b-1})|^2
        """
        if tempo_scale.size(1) < 3:
            return torch.tensor(0.0, device=tempo_scale.device)
        
        diff1 = tempo_scale[:, 1:] - tempo_scale[:, :-1]
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        return torch.mean(diff2 ** 2)

    def forward(self, predictions, targets):
        """
        predictions: dict of model outputs
        targets: dict of ground-truth expressive features
        """
        loss_timing = self.huber(predictions["delta_t"], targets["delta_t"])
        loss_velocity = F.mse_loss(predictions["velocity"], targets["velocity"])
        
        if "tempo_scale" in predictions:
            loss_smooth = self.compute_smoothness_loss(predictions["tempo_scale"])
        else:
            loss_smooth = torch.tensor(0.0, device=predictions["delta_t"].device)

        total_loss = (self.w_timing * loss_timing) + (self.w_velocity * loss_velocity) + (self.w_smooth * loss_smooth)
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "loss_timing": loss_timing.item(),
            "loss_velocity": loss_velocity.item(),
            "loss_smooth": loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else 0.0
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss_accum = 0.0
    timing_loss_accum = 0.0
    velocity_loss_accum = 0.0
    total_samples = 0

    for x, targets in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        predictions = model(x)
        loss, loss_components = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = x.size(0)
        total_loss_accum += loss_components["total_loss"] * batch_size
        timing_loss_accum += loss_components["loss_timing"] * batch_size
        velocity_loss_accum += loss_components["loss_velocity"] * batch_size
        total_samples += batch_size

    n = max(total_samples, 1)
    return total_loss_accum / n, timing_loss_accum / n, velocity_loss_accum / n

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss_accum = 0.0
    timing_loss_accum = 0.0
    velocity_loss_accum = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, targets in tqdm(dataloader, desc="Validation", leave=False):
            x = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            predictions = model(x)
            loss, loss_components = criterion(predictions, targets)

            batch_size = x.size(0)
            total_loss_accum += loss_components["total_loss"] * batch_size
            timing_loss_accum += loss_components["loss_timing"] * batch_size
            velocity_loss_accum += loss_components["loss_velocity"] * batch_size
            total_samples += batch_size

    n = max(total_samples, 1)
    return total_loss_accum / n, timing_loss_accum / n, velocity_loss_accum / n

def main():
    parser = argparse.ArgumentParser(description="Tenuto Expressive Score-to-Performance AI Training")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to preprocessed dataset")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "bigru"])
    parser.add_argument("--in_features", type=int, default=40, help="Input note feature vector dimension (10 or 40)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    print(f"[Tenuto] Initializing Model architecture: {args.model_type.upper()} ({args.in_features}D Features)...")
    model = build_model(model_name=args.model_type, in_features=args.in_features).to(device)
    criterion = TenutoLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader, val_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_timing, train_vel = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_timing, val_vel = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} (Timing: {train_timing:.5f}, Vel: {train_vel:.2f}) | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, filename=f"best_{args.model_type}_model.pth")

if __name__ == "__main__":
    main()
