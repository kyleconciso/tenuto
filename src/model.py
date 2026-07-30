import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for Sequential Note Tokens."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TenutoTransformer(nn.Module):
    r"""
    Non-Autoregressive Expression Model for Score-to-Performance AI Engine.
    
    Backbone: Bidirectional Transformer Encoder (~5M - 15M params)
    Output Heads:
      - Head 1: Beat Tempo Scale S(b) -> Rubato curve [0.5, 1.5]
      - Head 2: Dynamic Baseline V(b) -> Macro dynamics [0, 127]
      - Head 3: Micro Timing Shift \Delta t_i -> Finger asynchrony [-25ms, +25ms]
      - Head 4: Velocity Offset v_i -> Note velocity [0, 127]
      - Head 5: Articulation Scale d_i -> Key release duration multiplier (Softplus)
      - Head 6: Sustain Pedal State -> CC64 curve [0, 127]
    """
    def __init__(self, in_features: int = 40, d_model: int = 256, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1):
        super(TenutoTransformer, self).__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Parallel Output Heads
        self.head_tempo = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.head_dynamics = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.head_timing = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())
        self.head_velocity = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.head_articulation = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1), nn.Softplus())
        self.head_pedal = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

    def print_architecture_summary(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=================================================================")
        print("                TENUTO TRANSFORMER ARCHITECTURE                  ")
        print("=================================================================")
        print(f"  • Input Note Feature Dim:  {self.in_proj.in_features}D")
        print(f"  • Model Hidden Dim (d):    {self.in_proj.out_features}")
        print(f"  • Encoder Layers:          6 Layers (Multi-Head Self-Attention)")
        print(f"  • Attention Heads:         8 Heads")
        print(f"  • FFN Feedforward Dim:     1024")
        print(f"  • Output Parallel Heads:   6 Heads (Rubato, Dynamics, Timing, Velocity, Articulation, Pedal)")
        print(f"  • Total Trainable Params:  {total_params:,} (~{total_params/1e6:.2f} Million)")
        print("=================================================================\n")

    def forward(self, x, src_key_padding_mask=None):
        h = self.in_proj(x)
        h = self.pos_encoder(h)
        feat = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)

        tempo_scale = 0.5 + 1.0 * self.head_tempo(feat).squeeze(-1)       # [0.5, 1.5]
        dynamic_base = 127.0 * self.head_dynamics(feat).squeeze(-1)      # [0, 127]
        delta_t = 0.025 * self.head_timing(feat).squeeze(-1)              # [-0.025s, +0.025s]
        velocity = 127.0 * self.head_velocity(feat).squeeze(-1)           # [0, 127]
        articulation = self.head_articulation(feat).squeeze(-1)          # > 0
        pedal = 127.0 * self.head_pedal(feat).squeeze(-1)                 # [0, 127]

        return {
            "tempo_scale": tempo_scale,
            "dynamic_base": dynamic_base,
            "delta_t": delta_t,
            "velocity": velocity,
            "articulation": articulation,
            "pedal": pedal
        }

class TenutoBiGRU(nn.Module):
    """Lightweight 2-Layer Bidirectional GRU Prototype."""
    def __init__(self, in_features: int = 10, hidden_dim: int = 128, num_layers: int = 2):
        super(TenutoBiGRU, self).__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        d_out = hidden_dim * 2
        self.head_timing = nn.Sequential(nn.Linear(d_out, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.head_velocity = nn.Sequential(nn.Linear(d_out, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def print_architecture_summary(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=================================================================")
        print("                 TENUTO BiGRU ARCHITECTURE                       ")
        print("=================================================================")
        print(f"  • Total Trainable Params:  {total_params:,} (~{total_params/1e6:.2f} Million)")
        print("=================================================================\n")

    def forward(self, x):
        feat, _ = self.gru(x)
        delta_t = 0.025 * self.head_timing(feat).squeeze(-1)
        velocity = 127.0 * self.head_velocity(feat).squeeze(-1)
        return {"delta_t": delta_t, "velocity": velocity}

def build_model(model_name: str = "transformer", in_features: int = 40):
    if model_name.lower() == "transformer":
        return TenutoTransformer(in_features=in_features)
    elif model_name.lower() == "bigru":
        return TenutoBiGRU(in_features=in_features)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
