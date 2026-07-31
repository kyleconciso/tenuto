import argparse
import time
import os
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import flax
    from flax.training import train_state
    import optax
    from src.model_jax import TenutoTransformerJAX
    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    JAX_IMPORT_ERROR = str(e)

from src.utils import set_seed
from src.dataset import create_dataloaders

def huber_loss_jax(pred, target, delta=0.005):
    err = jnp.abs(pred - target)
    return jnp.where(err <= delta, 0.5 * (err ** 2), delta * (err - 0.5 * delta))

def compute_smoothness_loss_jax(tempo_scale):
    if tempo_scale.shape[1] < 3:
        return jnp.float32(0.0)
    diff1 = tempo_scale[:, 1:] - tempo_scale[:, :-1]
    diff2 = diff1[:, 1:] - diff1[:, :-1]
    return jnp.mean(jnp.square(diff2))

def compute_loss_jax(params, model, batch, w_timing=100.0, w_velocity=1.0, w_smooth=10.0, rng=None):
    x, targets = batch
    preds = model.apply({'params': params}, x, deterministic=True, rngs={'dropout': rng} if rng is not None else None)
    
    loss_timing = jnp.mean(huber_loss_jax(preds["delta_t"], targets["delta_t"]))
    loss_velocity = jnp.mean(jnp.square(preds["velocity"] - targets["velocity"]))
    
    loss_art = jnp.mean(jnp.square(preds["articulation"] - targets["articulation"])) if "articulation" in targets else 0.0
    loss_ped = jnp.mean(jnp.square(preds["pedal"] - targets["pedal"])) if "pedal" in targets else 0.0
    loss_smooth = compute_smoothness_loss_jax(preds["tempo_scale"]) if "tempo_scale" in preds else 0.0

    total_loss = (w_timing * loss_timing) + (w_velocity * loss_velocity) + loss_art + loss_ped + (w_smooth * loss_smooth)
    
    return total_loss, {
        "total_loss": total_loss,
        "loss_timing": loss_timing,
        "loss_velocity": loss_velocity,
        "loss_articulation": loss_art,
        "loss_pedal": loss_ped,
        "loss_smooth": loss_smooth
    }

def main(args=None):
    parser = argparse.ArgumentParser(description="Tenuto JAX/Flax Training on TPU v5e-1")
    parser.add_argument("--data_dir", type=str, default="./data/processed", help="Path to preprocessed dataset")
    parser.add_argument("--in_features", type=int, default=40, help="Input feature dimension")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for TPU training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parsed_args = parser.parse_args(args)
    args = parsed_args

    if not JAX_AVAILABLE:
        print(f"[TenutoJAX] ❌ JAX/Flax is not installed in this environment.")
        print(f"            Error: {JAX_IMPORT_ERROR}")
        print(f"            To train with JAX on TPU v5e-1, run in Colab TPU runtime or install dependencies:")
        print(f"            %pip install jax flax optax")
        return

    set_seed(args.seed)
    
    devices = jax.devices()
    print("=================================================================")
    print("                TENUTO JAX/FLAX TPU v5e-1 ENGINE                  ")
    print("=================================================================")
    print(f"  • Detected JAX Backend: {jax.default_backend().upper()}")
    print(f"  • Available Devices ({len(devices)}): {devices}")
    print("=================================================================\n")

    # Initialize model and parameters
    model = TenutoTransformerJAX(in_features=args.in_features)
    dummy_input = jnp.zeros((1, 256, args.in_features), dtype=jnp.float32)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, dummy_input, deterministic=True)['params']

    # Optax Optimizer with Cosine Decay
    total_steps = args.epochs * 100 # approximate step count fallback
    schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=total_steps)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    # JIT-compiled Train and Val Steps
    @jax.jit
    def train_step(state, batch, rng_key):
        grad_fn = jax.value_and_grad(compute_loss_jax, argnums=0, has_aux=True)
        (loss, loss_metrics), grads = grad_fn(state.params, model, batch, rng=rng_key)
        state = state.apply_gradients(grads=grads)
        return state, loss_metrics

    @jax.jit
    def val_step(params, batch):
        loss, loss_metrics = compute_loss_jax(params, model, batch)
        return loss_metrics

    train_loader, val_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    best_val_loss = float('inf')

    print(f"[TenutoJAX] Starting TPU v5e-1 training loop for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Training loop
        train_metrics_accum = {"total_loss": 0.0, "loss_timing": 0.0, "loss_velocity": 0.0, "loss_articulation": 0.0, "loss_pedal": 0.0}
        num_train_batches = 0

        for x_b, targets_b in train_loader:
            rng, step_rng = jax.random.split(rng)
            x_np = jnp.array(x_b.numpy())
            targets_np = {k: jnp.array(v.numpy()) for k, v in targets_b.items()}
            
            state, metrics = train_step(state, (x_np, targets_np), step_rng)
            for k in train_metrics_accum:
                train_metrics_accum[k] += float(metrics[k])
            num_train_batches += 1

        n_train = max(num_train_batches, 1)
        train_loss = train_metrics_accum["total_loss"] / n_train

        # Validation loop
        val_metrics_accum = {"total_loss": 0.0, "loss_timing": 0.0, "loss_velocity": 0.0, "loss_articulation": 0.0, "loss_pedal": 0.0}
        num_val_batches = 0

        for x_b, targets_b in val_loader:
            x_np = jnp.array(x_b.numpy())
            targets_np = {k: jnp.array(v.numpy()) for k, v in targets_b.items()}
            
            metrics = val_step(state.params, (x_np, targets_np))
            for k in val_metrics_accum:
                val_metrics_accum[k] += float(metrics[k])
            num_val_batches += 1

        n_val = max(num_val_batches, 1)
        val_loss = val_metrics_accum["total_loss"] / n_val

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Time: {elapsed:.2f}s (JAX TPU)")
        print(f"  ├── Train Loss : {train_loss:.4f} | Timing: {train_metrics_accum['loss_timing']/n_train:.5f}s | Vel: {train_metrics_accum['loss_velocity']/n_train:.2f}")
        print(f"  └── Val Loss   : {val_loss:.4f} | Timing: {val_metrics_accum['loss_timing']/n_val:.5f}s | Vel: {val_metrics_accum['loss_velocity']/n_val:.2f}")

        # Save Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_bytes = flax.serialization.to_bytes(state.params)
            local_ckpt = "checkpoints/best_transformer_jax.msgpack"
            with open(local_ckpt, "wb") as f:
                f.write(ckpt_bytes)
            print(f"[TenutoJAX] Saved best JAX model checkpoint (Val Loss: {best_val_loss:.4f})")

            gdrive_dir = "/content/drive/MyDrive/Tenuto/checkpoints"
            if os.path.exists("/content/drive/MyDrive"):
                os.makedirs(gdrive_dir, exist_ok=True)
                import shutil
                shutil.copy(local_ckpt, os.path.join(gdrive_dir, "best_transformer_jax.msgpack"))
                print(f"[TenutoJAX] Synced checkpoint to Google Drive: {gdrive_dir}")

if __name__ == "__main__":
    main()
