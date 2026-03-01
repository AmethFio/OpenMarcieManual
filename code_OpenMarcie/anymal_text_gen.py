"""
AnyMAL-like Multimodal Text Generation

Based on "AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model"
(Moon et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple


# =============================================================================
# Modality Encoders (reused from Perceiver IO with modifications)
# =============================================================================

class IMUEncoder(nn.Module):
    """Encoder for IMU sensor data (accelerometer + gyroscope)."""
    def __init__(self, input_size, hidden_dim=128, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.proj(x)


class AudioEncoder(nn.Module):
    """Encoder for audio embeddings."""
    def __init__(self, feature_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)


class VideoEncoder(nn.Module):
    """Encoder for video embeddings."""
    def __init__(self, feature_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class MagnetometerEncoder(nn.Module):
    """Encoder for magnetometer data (3-axis magnetic field)."""
    def __init__(self, input_size=3, hidden_dim=64, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.proj(x)


class BarometerEncoder(nn.Module):
    """Encoder for barometer (pressure) data."""
    def __init__(self, input_size=1, hidden_dim=32, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return self.proj(x)


class TemperatureEncoder(nn.Module):
    """Encoder for temperature sensor data."""
    def __init__(self, input_size=1, hidden_dim=32, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return self.proj(x)


class SpectrometerEncoder(nn.Module):
    """Encoder for spectrometer data (spectral measurements)."""
    def __init__(self, num_channels, hidden_dim=128, output_dim=256):
        super().__init__()
        self.temporal_proj = nn.Linear(num_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.temporal_proj(x)
        x, _ = self.lstm(x)
        x = self.proj(x)
        return self.norm(x)


class ThermalEncoder(nn.Module):
    """Encoder for thermal camera/sensor data."""
    def __init__(self, input_shape, hidden_dim=128, output_dim=256):
        super().__init__()
        self.input_shape = input_shape
        
        if len(input_shape) == 2:  # 2D thermal image
            h, w = input_shape
            self.is_2d = True
            self.conv_net = nn.Sequential(
                nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(hidden_dim * 16, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:  # 1D thermal array
            self.is_2d = False
            num_sensors = input_shape[0]
            self.net = nn.Sequential(
                nn.Linear(num_sensors, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            self.lstm = nn.LSTM(output_dim, output_dim // 2, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        if self.is_2d:
            batch_size, seq_len, h, w = x.shape
            x = x.view(batch_size * seq_len, 1, h, w)
            x = self.conv_net(x)
            x = x.view(batch_size, seq_len, -1)
        else:
            x = self.net(x)
            x, _ = self.lstm(x)
        return x


# =============================================================================
# AnyMAL Projection Networks
# =============================================================================

class ModalityProjector(nn.Module):
    """
    Lightweight MLP projector to map modality encodings to LLM embedding space.
    Following AnyMAL's approach of using simple projectors.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.proj(x)


class ModalityTokenPooler(nn.Module):
    """
    Pool variable-length modality sequences into fixed number of tokens.
    Uses learned queries (similar to Q-Former/Perceiver).
    """
    def __init__(self, input_dim, num_tokens=8, num_heads=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.queries = nn.Parameter(torch.randn(1, num_tokens, input_dim) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            Pooled tokens [batch, num_tokens, dim]
        """
        batch_size = x.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)
        
        pooled, _ = self.cross_attn(queries, x, x)
        return self.norm(pooled + queries)


# =============================================================================
# Transformer Decoder (LLM) Components
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding."""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
        
    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Helper for rotary embedding."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block with pre-norm."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder (LLM) for text generation.
    Simplified version inspired by LLaMA architecture.
    """
    def __init__(
        self,
        vocab_size,
        dim=512,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4,
        max_seq_len=512,
        dropout=0.1,
        pad_token_id=0
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final norm and output projection
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, input_ids, attention_mask=None, input_embeds=None):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            input_embeds: Optional pre-computed embeddings [batch, seq_len, dim]
                         (used when prepending modality tokens)
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        if input_embeds is None:
            x = self.token_embedding(input_ids)
        else:
            x = input_embeds
            
        for block in self.blocks:
            x = block(x, attention_mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


# =============================================================================
# AnyMAL Multimodal Text Generator
# =============================================================================

class AnyMALTextGenerator(nn.Module):
    """
    AnyMAL-like architecture for multimodal text generation.
    
    Pipeline:
    1. Encode each modality using specialized encoders
    2. Pool encodings into fixed number of tokens per modality
    3. Project to LLM embedding space
    4. Concatenate modality tokens with text tokens
    5. Generate text autoregressively
    
    Supports: IMU, Audio, Video, Magnetometer, Barometer, Temperature, 
              Spectrometer, Thermal
    """
    def __init__(
        self,
        vocab_size: int,
        llm_dim: int = 512,
        encoder_dim: int = 256,
        num_modality_tokens: int = 8,
        llm_num_layers: int = 6,
        llm_num_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # Modality configurations
        imu_input_size: int = 6,
        audio_feature_dim: int = 128,
        video_feature_dim: int = 512,
        magnetometer_input_size: Optional[int] = 3,
        barometer_input_size: Optional[int] = 1,
        temperature_input_size: Optional[int] = 1,
        spectrometer_channels: Optional[int] = None,
        thermal_shape: Optional[tuple] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.llm_dim = llm_dim
        self.encoder_dim = encoder_dim
        self.num_modality_tokens = num_modality_tokens
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Track enabled modalities
        self.modality_names = ['imu', 'audio', 'video']
        self.use_magnetometer = magnetometer_input_size is not None
        self.use_barometer = barometer_input_size is not None
        self.use_temperature = temperature_input_size is not None
        self.use_spectrometer = spectrometer_channels is not None
        self.use_thermal = thermal_shape is not None
        
        # =====================================================================
        # Modality Encoders
        # =====================================================================
        
        # Core modalities (always enabled)
        self.imu_encoder = IMUEncoder(imu_input_size, output_dim=encoder_dim)
        self.audio_encoder = AudioEncoder(audio_feature_dim, output_dim=encoder_dim)
        self.video_encoder = VideoEncoder(video_feature_dim, output_dim=encoder_dim)
        
        # Optional modalities
        if self.use_magnetometer:
            self.magnetometer_encoder = MagnetometerEncoder(
                magnetometer_input_size, output_dim=encoder_dim
            )
            self.modality_names.append('magnetometer')
            
        if self.use_barometer:
            self.barometer_encoder = BarometerEncoder(
                barometer_input_size, output_dim=encoder_dim
            )
            self.modality_names.append('barometer')
            
        if self.use_temperature:
            self.temperature_encoder = TemperatureEncoder(
                temperature_input_size, output_dim=encoder_dim
            )
            self.modality_names.append('temperature')
            
        if self.use_spectrometer:
            self.spectrometer_encoder = SpectrometerEncoder(
                spectrometer_channels, output_dim=encoder_dim
            )
            self.modality_names.append('spectrometer')
            
        if self.use_thermal:
            self.thermal_encoder = ThermalEncoder(
                thermal_shape, output_dim=encoder_dim
            )
            self.modality_names.append('thermal')
        
        # =====================================================================
        # Token Poolers (convert variable-length to fixed tokens)
        # =====================================================================
        
        self.poolers = nn.ModuleDict({
            name: ModalityTokenPooler(encoder_dim, num_modality_tokens)
            for name in self.modality_names
        })
        
        # =====================================================================
        # Projectors (map to LLM embedding space)
        # =====================================================================
        
        self.projectors = nn.ModuleDict({
            name: ModalityProjector(encoder_dim, llm_dim)
            for name in self.modality_names
        })
        
        # =====================================================================
        # Special tokens for modality boundaries
        # =====================================================================
        
        self.modality_start_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)
            for name in self.modality_names
        })
        
        self.modality_end_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)
            for name in self.modality_names
        })
        
        # =====================================================================
        # LLM Decoder
        # =====================================================================
        
        self.llm = TransformerDecoder(
            vocab_size=vocab_size,
            dim=llm_dim,
            num_layers=llm_num_layers,
            num_heads=llm_num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
    def encode_modality(self, name: str, data: torch.Tensor) -> torch.Tensor:
        """Encode a single modality and project to LLM space."""
        # Get encoder
        encoder = getattr(self, f'{name}_encoder')
        encoded = encoder(data)  # [batch, seq, encoder_dim]
        
        # Pool to fixed tokens
        pooled = self.poolers[name](encoded)  # [batch, num_tokens, encoder_dim]
        
        # Project to LLM space
        projected = self.projectors[name](pooled)  # [batch, num_tokens, llm_dim]
        
        return projected
    
    def prepare_multimodal_input(
        self,
        text_ids: torch.Tensor,
        imu: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        magnetometer: Optional[torch.Tensor] = None,
        barometer: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        spectrometer: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare multimodal input by encoding modalities and interleaving with text.
        
        Following AnyMAL: [MOD1_START][mod1_tokens][MOD1_END]...[text_tokens]
        
        Returns:
            input_embeds: Combined embeddings [batch, total_seq, llm_dim]
            attention_mask: Attention mask [batch, total_seq]
        """
        batch_size = text_ids.shape[0]
        device = text_ids.device
        
        modality_embeds = []
        
        # Collect modality data
        modality_data = {
            'imu': imu,
            'audio': audio,
            'video': video,
            'magnetometer': magnetometer if self.use_magnetometer else None,
            'barometer': barometer if self.use_barometer else None,
            'temperature': temperature if self.use_temperature else None,
            'spectrometer': spectrometer if self.use_spectrometer else None,
            'thermal': thermal if self.use_thermal else None
        }
        
        # Encode each available modality
        for name in self.modality_names:
            data = modality_data.get(name)
            if data is not None:
                # Start token
                start_token = self.modality_start_tokens[name].expand(batch_size, -1, -1)
                
                # Encoded modality tokens
                mod_tokens = self.encode_modality(name, data)
                
                # End token
                end_token = self.modality_end_tokens[name].expand(batch_size, -1, -1)
                
                modality_embeds.append(start_token)
                modality_embeds.append(mod_tokens)
                modality_embeds.append(end_token)
        
        # Get text embeddings
        text_embeds = self.llm.token_embedding(text_ids)  # [batch, text_len, llm_dim]
        
        # Concatenate: [modality_tokens] + [text_tokens]
        if modality_embeds:
            all_modality_embeds = torch.cat(modality_embeds, dim=1)
            input_embeds = torch.cat([all_modality_embeds, text_embeds], dim=1)
        else:
            input_embeds = text_embeds
        
        # Create attention mask (all ones for now, could mask padding)
        attention_mask = torch.ones(batch_size, input_embeds.shape[1], device=device)
        
        return input_embeds, attention_mask
    
    def forward(
        self,
        text_ids: torch.Tensor,
        imu: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        magnetometer: Optional[torch.Tensor] = None,
        barometer: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        spectrometer: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            text_ids: Input token IDs [batch, seq_len]
            imu, audio, video, etc.: Modality inputs
            labels: Target token IDs for loss computation [batch, seq_len]
            
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Prepare multimodal input
        input_embeds, attention_mask = self.prepare_multimodal_input(
            text_ids, imu, audio, video,
            magnetometer, barometer, temperature, spectrometer, thermal
        )
        
        # Forward through LLM
        logits = self.llm(input_ids=None, input_embeds=input_embeds, attention_mask=attention_mask)
        
        output = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Need to align labels with logits (accounting for modality tokens)
            num_modality_tokens = input_embeds.shape[1] - text_ids.shape[1]
            
            # Only compute loss on text portion
            text_logits = logits[:, num_modality_tokens:, :]
            
            # Shift for next-token prediction
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
            output['loss'] = loss
            
        return output
    
    @torch.no_grad()
    def generate(
        self,
        imu: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        magnetometer: Optional[torch.Tensor] = None,
        barometer: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        spectrometer: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature_sampling: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text from multimodal inputs.
        
        Args:
            imu, audio, video, etc.: Modality inputs
            prompt_ids: Optional text prompt [batch, prompt_len]
            max_length: Maximum generation length
            temperature_sampling: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs [batch, generated_len]
        """
        self.eval()
        
        # Determine batch size from first available modality
        batch_size = None
        for data in [imu, audio, video, magnetometer, barometer, temperature, spectrometer, thermal]:
            if data is not None:
                batch_size = data.shape[0]
                device = data.device
                break
        
        if batch_size is None:
            raise ValueError("At least one modality must be provided")
        
        # Initialize with BOS token if no prompt
        if prompt_ids is None:
            current_ids = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
        else:
            current_ids = prompt_ids
        
        # Encode modalities once
        modality_embeds = []
        modality_data = {
            'imu': imu, 'audio': audio, 'video': video,
            'magnetometer': magnetometer if self.use_magnetometer else None,
            'barometer': barometer if self.use_barometer else None,
            'temperature': temperature if self.use_temperature else None,
            'spectrometer': spectrometer if self.use_spectrometer else None,
            'thermal': thermal if self.use_thermal else None
        }
        
        for name in self.modality_names:
            data = modality_data.get(name)
            if data is not None:
                start_token = self.modality_start_tokens[name].expand(batch_size, -1, -1)
                mod_tokens = self.encode_modality(name, data)
                end_token = self.modality_end_tokens[name].expand(batch_size, -1, -1)
                modality_embeds.extend([start_token, mod_tokens, end_token])
        
        if modality_embeds:
            modality_prefix = torch.cat(modality_embeds, dim=1)
        else:
            modality_prefix = None
        
        # Autoregressive generation
        generated_ids = []
        
        for _ in range(max_length):
            # Get text embeddings
            text_embeds = self.llm.token_embedding(current_ids)
            
            # Combine with modality prefix
            if modality_prefix is not None:
                input_embeds = torch.cat([modality_prefix, text_embeds], dim=1)
            else:
                input_embeds = text_embeds
            
            # Forward pass
            logits = self.llm(input_ids=None, input_embeds=input_embeds)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature_sampling
            
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.eos_token_id).all():
                break
        
        return torch.cat(generated_ids, dim=1)


# =============================================================================
# Training Script
# =============================================================================

def train_anymal(
    vocab_size: int = 32000,
    use_magnetometer: bool = False,
    use_barometer: bool = False,
    use_temperature: bool = False,
    use_spectrometer: bool = False,
    use_thermal: bool = False
):
    """
    Training function for AnyMAL multimodal text generator.
    
    Args:
        vocab_size: Size of vocabulary (e.g., 32000 for LLaMA tokenizer)
        use_*: Enable optional modalities
    """
    from fixedwindowloader import FixedWindowDataset
    
    # Configuration
    base_dir = r"E:\precomputed_data"
    save_path = "anymal_best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 100
    llm_dim = 512
    encoder_dim = 256
    num_modality_tokens = 8
    llm_num_layers = 6
    
    # Optional modality configs
    magnetometer_input_size = 3 if use_magnetometer else None
    barometer_input_size = 1 if use_barometer else None
    temperature_input_size = 1 if use_temperature else None
    spectrometer_channels = 128 if use_spectrometer else None
    thermal_shape = (32,) if use_thermal else None
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FixedWindowDataset(
        data_dir=os.path.join(base_dir, "train"),
        imu_window_size=100,
        audio_window_size=250,
        video_window_size=4,
        stride=50
    )
    val_dataset = FixedWindowDataset(
        data_dir=os.path.join(base_dir, "val"),
        imu_window_size=100,
        audio_window_size=250,
        video_window_size=4,
        stride=50
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input shapes
    sample = next(iter(train_loader))
    imu_input_size = sample["imu"].shape[-1]
    audio_feature_dim = sample["audio"].shape[1]
    video_feature_dim = sample["video"].shape[-1]
    
    print(f"\nInput shapes:")
    print(f"  IMU: {imu_input_size}")
    print(f"  Audio: {audio_feature_dim}")
    print(f"  Video: {video_feature_dim}")
    
    print(f"\nEnabled modalities:")
    print(f"  Core: IMU, Audio, Video")
    print(f"  Magnetometer: {'Yes' if use_magnetometer else 'No'}")
    print(f"  Barometer: {'Yes' if use_barometer else 'No'}")
    print(f"  Temperature: {'Yes' if use_temperature else 'No'}")
    print(f"  Spectrometer: {'Yes' if use_spectrometer else 'No'}")
    print(f"  Thermal: {'Yes' if use_thermal else 'No'}")
    
    # Initialize model
    model = AnyMALTextGenerator(
        vocab_size=vocab_size,
        llm_dim=llm_dim,
        encoder_dim=encoder_dim,
        num_modality_tokens=num_modality_tokens,
        llm_num_layers=llm_num_layers,
        llm_num_heads=8,
        max_seq_len=512,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        imu_input_size=imu_input_size,
        audio_feature_dim=audio_feature_dim,
        video_feature_dim=video_feature_dim,
        magnetometer_input_size=magnetometer_input_size,
        barometer_input_size=barometer_input_size,
        temperature_input_size=temperature_input_size,
        spectrometer_channels=spectrometer_channels,
        thermal_shape=thermal_shape
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            imu = batch["imu"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            
            # Note: You need to add text_ids and labels to your dataset
            # For now, using sentence_embedding as a placeholder
            # In practice, you'd have tokenized text
            text_ids = batch.get("text_ids")
            if text_ids is None:
                # Placeholder: create dummy text tokens
                text_ids = torch.randint(3, vocab_size, (imu.shape[0], 50), device=device)
            else:
                text_ids = text_ids.to(device)
            
            labels = text_ids.clone()
            
            # Extract optional modalities
            magnetometer = batch.get("magnetometer")
            barometer = batch.get("barometer")
            temperature = batch.get("temperature")
            spectrometer = batch.get("spectrometer")
            thermal = batch.get("thermal")
            
            if magnetometer is not None:
                magnetometer = magnetometer.to(device)
            if barometer is not None:
                barometer = barometer.to(device)
            if temperature is not None:
                temperature = temperature.to(device)
            if spectrometer is not None:
                spectrometer = spectrometer.to(device)
            if thermal is not None:
                thermal = thermal.to(device)
            
            optimizer.zero_grad()
            
            output = model(
                text_ids=text_ids,
                imu=imu,
                audio=audio,
                video=video,
                magnetometer=magnetometer,
                barometer=barometer,
                temperature=temperature,
                spectrometer=spectrometer,
                thermal=thermal,
                labels=labels
            )
            
            loss = output['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imu = batch["imu"].to(device)
                audio = batch["audio"].to(device)
                video = batch["video"].to(device)
                
                text_ids = batch.get("text_ids")
                if text_ids is None:
                    text_ids = torch.randint(3, vocab_size, (imu.shape[0], 50), device=device)
                else:
                    text_ids = text_ids.to(device)
                
                labels = text_ids.clone()
                
                magnetometer = batch.get("magnetometer")
                barometer = batch.get("barometer")
                temperature = batch.get("temperature")
                spectrometer = batch.get("spectrometer")
                thermal = batch.get("thermal")
                
                if magnetometer is not None:
                    magnetometer = magnetometer.to(device)
                if barometer is not None:
                    barometer = barometer.to(device)
                if temperature is not None:
                    temperature = temperature.to(device)
                if spectrometer is not None:
                    spectrometer = spectrometer.to(device)
                if thermal is not None:
                    thermal = thermal.to(device)
                
                output = model(
                    text_ids=text_ids,
                    imu=imu,
                    audio=audio,
                    video=video,
                    magnetometer=magnetometer,
                    barometer=barometer,
                    temperature=temperature,
                    spectrometer=spectrometer,
                    thermal=thermal,
                    labels=labels
                )
                
                total_val_loss += output['loss'].item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'llm_dim': llm_dim,
                    'encoder_dim': encoder_dim,
                    'num_modality_tokens': num_modality_tokens,
                    'llm_num_layers': llm_num_layers,
                    'imu_input_size': imu_input_size,
                    'audio_feature_dim': audio_feature_dim,
                    'video_feature_dim': video_feature_dim,
                    'magnetometer_input_size': magnetometer_input_size,
                    'barometer_input_size': barometer_input_size,
                    'temperature_input_size': temperature_input_size,
                    'spectrometer_channels': spectrometer_channels,
                    'thermal_shape': thermal_shape
                }
            }, save_path)
            print(f"  -> New best model saved! (Val Loss: {best_val_loss:.4f})")
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    return model


# =============================================================================
# Inference Example
# =============================================================================

def generate_text_from_sensors(
    checkpoint_path: str,
    imu: torch.Tensor,
    audio: torch.Tensor,
    video: torch.Tensor,
    tokenizer=None,
    max_length: int = 100,
    **kwargs
):
    """
    Generate text description from sensor inputs.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        imu, audio, video: Sensor inputs
        tokenizer: Tokenizer for decoding (if None, returns token IDs)
        max_length: Maximum generation length
        **kwargs: Additional modalities (magnetometer, barometer, etc.)
        
    Returns:
        Generated text or token IDs
    """
    device = imu.device
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = AnyMALTextGenerator(
        vocab_size=config['vocab_size'],
        llm_dim=config['llm_dim'],
        encoder_dim=config['encoder_dim'],
        num_modality_tokens=config['num_modality_tokens'],
        llm_num_layers=config['llm_num_layers'],
        llm_num_heads=8,
        imu_input_size=config['imu_input_size'],
        audio_feature_dim=config['audio_feature_dim'],
        video_feature_dim=config['video_feature_dim'],
        magnetometer_input_size=config.get('magnetometer_input_size'),
        barometer_input_size=config.get('barometer_input_size'),
        temperature_input_size=config.get('temperature_input_size'),
        spectrometer_channels=config.get('spectrometer_channels'),
        thermal_shape=config.get('thermal_shape')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate
    generated_ids = model.generate(
        imu=imu,
        audio=audio,
        video=video,
        max_length=max_length,
        **kwargs
    )
    
    if tokenizer is not None:
        return tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    
    return generated_ids


if __name__ == "__main__":
    # Train with core modalities only
    train_anymal(vocab_size=32000)
    
    # Or train with all modalities:
    # train_anymal(
    #     vocab_size=32000,
    #     use_magnetometer=True,
    #     use_barometer=True,
    #     use_temperature=True,
    #     use_spectrometer=True,
    #     use_thermal=True
    # )
