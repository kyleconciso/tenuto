import math
import jax
import jax.numpy as jnp
import flax.linen as nn

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding in JAX/Flax."""
    d_model: int
    max_len: int = 15000

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * (-math.log(10000.0) / self.d_model))
        
        pe_even = jnp.sin(position * div_term)
        pe_odd = jnp.cos(position * div_term)
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(pe_even)
        pe = pe.at[:, 1::2].set(pe_odd)
        
        pe = jnp.expand_dims(pe[:seq_len], axis=0)
        return x + pe

class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder Block in Flax with bfloat16 mixed precision support."""
    d_model: int = 256
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Self-Attention Branch
        norm_x = nn.LayerNorm(dtype=self.dtype)(x)
        attn_out = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.d_model, dtype=self.dtype)(norm_x)
        attn_out = nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=deterministic)
        x = x + attn_out

        # Feed-Forward Branch
        norm_x2 = nn.LayerNorm(dtype=self.dtype)(x)
        ffn = nn.Dense(self.dim_feedforward, dtype=self.dtype)(norm_x2)
        ffn = nn.relu(ffn)
        ffn = nn.Dropout(rate=self.dropout_rate)(ffn, deterministic=deterministic)
        ffn = nn.Dense(self.d_model, dtype=self.dtype)(ffn)
        ffn = nn.Dropout(rate=self.dropout_rate)(ffn, deterministic=deterministic)
        
        return x + ffn

class TenutoTransformerJAX(nn.Module):
    r"""
    JAX / Flax Implementation of Tenuto Non-Autoregressive Score-to-Performance Engine.
    Optimized for TPU v5e-1 Native XLA Acceleration with bfloat16 Mixed-Precision.
    """
    in_features: int = 40
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout_rate: float = 0.1
    use_bfloat16: bool = True

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        x = x.astype(dtype)

        # Projection & Positional Encoding
        h = nn.Dense(self.d_model, dtype=dtype)(x)
        h = SinusoidalPositionalEncoding(d_model=self.d_model)(h)

        # Transformer Encoder Stack (bfloat16 MXU Execution)
        for _ in range(self.num_layers):
            h = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout_rate=self.dropout_rate,
                dtype=dtype
            )(h, deterministic=deterministic)

        h = nn.LayerNorm(dtype=dtype)(h)

        # 6 Parallel Output Heads (float32 precision accumulation for stability)
        h_f32 = h.astype(jnp.float32)
        tempo_scale = 0.5 + 1.0 * jax.nn.sigmoid(nn.Dense(1)(nn.relu(nn.Dense(128)(h_f32))))[..., 0]
        dynamic_base = 127.0 * jax.nn.sigmoid(nn.Dense(1)(nn.relu(nn.Dense(128)(h_f32))))[..., 0]
        delta_t = 0.025 * jax.nn.tanh(nn.Dense(1)(nn.relu(nn.Dense(128)(h_f32))))[..., 0]
        velocity = 127.0 * jax.nn.sigmoid(nn.Dense(1)(nn.relu(nn.Dense(128)(h_f32))))[..., 0]
        articulation = jax.nn.softplus(nn.Dense(1)(nn.relu(nn.Dense(128)(h_f32))))[..., 0]
        pedal = 127.0 * jax.nn.sigmoid(nn.Dense(1)(nn.relu(nn.Dense(128)(h_f32))))[..., 0]

        return {
            "tempo_scale": tempo_scale,
            "dynamic_base": dynamic_base,
            "delta_t": delta_t,
            "velocity": velocity,
            "articulation": articulation,
            "pedal": pedal
        }
