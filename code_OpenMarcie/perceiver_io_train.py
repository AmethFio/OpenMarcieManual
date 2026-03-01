"""
Perceiver IO Multimodal Fusion Classification

Based on "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
(Jaegle et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset


# =============================================================================
# Perceiver IO Building Blocks
# =============================================================================

class FourierPositionalEncoding(nn.Module):
    """Fourier feature positional encoding for continuous inputs."""
    def __init__(self, input_dim, num_bands=64, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.input_dim = input_dim
        # Output dim = input_dim * (2 * num_bands + 1)
        self.output_dim = input_dim * (2 * num_bands + 1)
        
    def forward(self, x):
        """
        Args:
            x: Input positions of shape [batch, seq_len, input_dim]
        Returns:
            Fourier features of shape [batch, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate frequency bands
        freqs = torch.linspace(1.0, self.max_freq, self.num_bands, device=x.device)
        freqs = freqs.view(1, 1, 1, -1)  # [1, 1, 1, num_bands]
        
        x_expanded = x.unsqueeze(-1)  # [batch, seq_len, input_dim, 1]
        
        # Compute sin and cos features
        x_freq = x_expanded * freqs * 2 * math.pi  # [batch, seq_len, input_dim, num_bands]
        sin_features = torch.sin(x_freq)
        cos_features = torch.cos(x_freq)
        
        # Concatenate original input with sin/cos features
        features = torch.cat([x, sin_features.flatten(-2), cos_features.flatten(-2)], dim=-1)
        return features


class CrossAttention(nn.Module):
    """Cross-attention module where queries attend to key-value pairs."""
    def __init__(self, query_dim, kv_dim, num_heads=4, head_dim=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        inner_dim = num_heads * head_dim
        
        self.q_proj = nn.Linear(query_dim, inner_dim)
        self.k_proj = nn.Linear(kv_dim, inner_dim)
        self.v_proj = nn.Linear(kv_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        
    def forward(self, query, kv, attention_mask=None):
        """
        Args:
            query: [batch, num_queries, query_dim]
            kv: [batch, num_kv, kv_dim]
            attention_mask: Optional mask [batch, num_queries, num_kv]
        Returns:
            Output of shape [batch, num_queries, query_dim]
        """
        batch_size = query.shape[0]
        
        # Normalize inputs
        query = self.norm_q(query)
        kv = self.norm_kv(kv)
        
        # Project to multi-head
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.out_proj(out)


class SelfAttention(nn.Module):
    """Standard self-attention module."""
    def __init__(self, dim, num_heads=4, head_dim=64, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(dim, dim, num_heads, head_dim, dropout)
        
    def forward(self, x):
        return self.cross_attn(x, x)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Single Perceiver processing block with cross-attention + latent self-attention."""
    def __init__(self, latent_dim, input_dim, num_heads=4, head_dim=64, 
                 num_self_attn_layers=6, dropout=0.1):
        super().__init__()
        
        # Cross-attention from latent to input
        self.cross_attn = CrossAttention(latent_dim, input_dim, num_heads, head_dim, dropout)
        self.cross_ff = FeedForward(latent_dim, dropout=dropout)
        
        # Self-attention layers in latent space
        self.self_attn_layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(latent_dim, num_heads, head_dim, dropout),
                FeedForward(latent_dim, dropout=dropout)
            ]) for _ in range(num_self_attn_layers)
        ])
        
    def forward(self, latent, inputs):
        """
        Args:
            latent: [batch, num_latents, latent_dim]
            inputs: [batch, num_inputs, input_dim]
        Returns:
            Updated latent array [batch, num_latents, latent_dim]
        """
        # Cross-attention
        latent = latent + self.cross_attn(latent, inputs)
        latent = latent + self.cross_ff(latent)
        
        # Self-attention in latent space
        for self_attn, ff in self.self_attn_layers:
            latent = latent + self_attn(latent)
            latent = latent + ff(latent)
        
        return latent


# =============================================================================
# Modality-Specific Encoders
# =============================================================================

class IMUEncoder(nn.Module):
    """Encoder for IMU sensor data using ConvLSTM."""
    def __init__(self, input_size, hidden_dim=128, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            Features [batch, seq_len, output_dim]
        """
        x = x.permute(0, 2, 1)  # [batch, input_size, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
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
        """
        Args:
            x: [batch, feature_dim, seq_len]
        Returns:
            Features [batch, seq_len, output_dim]
        """
        x = x.permute(0, 2, 1)  # [batch, seq_len, feature_dim]
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
        """
        Args:
            x: [batch, seq_len, feature_dim]
        Returns:
            Features [batch, seq_len, output_dim]
        """
        return self.net(x)


class MagnetometerEncoder(nn.Module):
    """Encoder for magnetometer data (3-axis magnetic field measurements)."""
    def __init__(self, input_size=3, hidden_dim=64, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 3] (x, y, z magnetic field)
        Returns:
            Features [batch, seq_len, output_dim]
        """
        x = x.permute(0, 2, 1)  # [batch, 3, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
        x, _ = self.lstm(x)
        return self.proj(x)


class BarometerEncoder(nn.Module):
    """Encoder for barometer (pressure) data - scalar time series."""
    def __init__(self, input_size=1, hidden_dim=32, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 1] or [batch, seq_len] (pressure values)
        Returns:
            Features [batch, seq_len, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = x.permute(0, 2, 1)  # [batch, 1, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
        x, _ = self.gru(x)
        return self.proj(x)


class TemperatureEncoder(nn.Module):
    """Encoder for temperature sensor data - scalar time series."""
    def __init__(self, input_size=1, hidden_dim=32, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 1] or [batch, seq_len] (temperature values)
        Returns:
            Features [batch, seq_len, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = x.permute(0, 2, 1)  # [batch, 1, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
        x, _ = self.gru(x)
        return self.proj(x)


class SpectrometerEncoder(nn.Module):
    """Encoder for spectrometer data (wavelength/frequency spectrum measurements)."""
    def __init__(self, num_channels, hidden_dim=128, output_dim=256):
        super().__init__()
        # 1D CNN for spectral features across wavelength bands
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Temporal processing
        self.temporal_proj = nn.Linear(num_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, num_channels] (spectral measurements over time)
        Returns:
            Features [batch, seq_len, output_dim]
        """
        batch_size, seq_len, num_channels = x.shape
        
        # Process each timestep's spectrum
        x = self.temporal_proj(x)  # [batch, seq_len, hidden_dim]
        x, _ = self.lstm(x)
        x = self.proj(x)
        return self.norm(x)


class ThermalEncoder(nn.Module):
    """Encoder for thermal camera data (2D heat maps or 1D thermal array)."""
    def __init__(self, input_shape, hidden_dim=128, output_dim=256):
        super().__init__()
        self.input_shape = input_shape
        
        if len(input_shape) == 2:  # 2D thermal image [H, W]
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
        else:  # 1D thermal array [num_sensors]
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
        """
        Args:
            x: [batch, seq_len, H, W] for 2D or [batch, seq_len, num_sensors] for 1D
        Returns:
            Features [batch, seq_len, output_dim]
        """
        if self.is_2d:
            batch_size, seq_len, h, w = x.shape
            x = x.view(batch_size * seq_len, 1, h, w)  # [batch*seq, 1, H, W]
            x = self.conv_net(x)  # [batch*seq, output_dim]
            x = x.view(batch_size, seq_len, -1)  # [batch, seq_len, output_dim]
        else:
            x = self.net(x)  # [batch, seq_len, output_dim]
            x, _ = self.lstm(x)
        return x


# =============================================================================
# Perceiver IO Multimodal Fusion Classifier
# =============================================================================

class PerceiverIOFusionClassifier(nn.Module):
    """
    Perceiver IO architecture for multimodal fusion classification.
    
    Supports modalities:
    - IMU (accelerometer + gyroscope)
    - Audio embeddings
    - Video embeddings
    - Magnetometer (3-axis magnetic field)
    - Barometer (pressure)
    - Temperature
    - Spectrometer (spectral measurements)
    - Thermal (thermal camera/array)
    
    Architecture:
    1. Encode each modality to a common dimension
    2. Concatenate all modality features into a single byte array
    3. Cross-attend from learned latent array to byte array
    4. Process in latent space with self-attention
    5. Decode classification output via output query
    """
    def __init__(
        self,
        imu_input_size,
        audio_shape,
        video_shape,
        num_classes=13,
        latent_dim=256,
        num_latents=64,
        num_perceiver_blocks=2,
        num_self_attn_per_block=4,
        num_heads=4,
        head_dim=64,
        dropout=0.1,
        # Optional modality configurations (set to None to disable)
        magnetometer_input_size=3,
        barometer_input_size=1,
        temperature_input_size=1,
        spectrometer_channels=None,
        thermal_shape=None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        
        # Track which modalities are enabled
        self.use_magnetometer = magnetometer_input_size is not None
        self.use_barometer = barometer_input_size is not None
        self.use_temperature = temperature_input_size is not None
        self.use_spectrometer = spectrometer_channels is not None
        self.use_thermal = thermal_shape is not None
        
        # Core modality encoders (always enabled)
        self.imu_encoder = IMUEncoder(imu_input_size, output_dim=latent_dim)
        self.audio_encoder = AudioEncoder(audio_shape[0], output_dim=latent_dim)
        self.video_encoder = VideoEncoder(video_shape[1], output_dim=latent_dim)
        
        # Core modality type embeddings (learnable)
        self.imu_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.audio_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.video_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        # Optional modality encoders
        if self.use_magnetometer:
            self.magnetometer_encoder = MagnetometerEncoder(
                input_size=magnetometer_input_size, output_dim=latent_dim
            )
            self.magnetometer_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        if self.use_barometer:
            self.barometer_encoder = BarometerEncoder(
                input_size=barometer_input_size, output_dim=latent_dim
            )
            self.barometer_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        if self.use_temperature:
            self.temperature_encoder = TemperatureEncoder(
                input_size=temperature_input_size, output_dim=latent_dim
            )
            self.temperature_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        if self.use_spectrometer:
            self.spectrometer_encoder = SpectrometerEncoder(
                num_channels=spectrometer_channels, output_dim=latent_dim
            )
            self.spectrometer_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        if self.use_thermal:
            self.thermal_encoder = ThermalEncoder(
                input_shape=thermal_shape, output_dim=latent_dim
            )
            self.thermal_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        # Learned latent array
        self.latent_array = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)
        
        # Perceiver blocks (cross-attention + self-attention)
        self.perceiver_blocks = nn.ModuleList([
            PerceiverBlock(
                latent_dim=latent_dim,
                input_dim=latent_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_self_attn_layers=num_self_attn_per_block,
                dropout=dropout
            ) for _ in range(num_perceiver_blocks)
        ])
        
        # Output query for classification (learnable)
        self.output_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        # Output cross-attention (query attends to latent)
        self.output_cross_attn = CrossAttention(
            latent_dim, latent_dim, num_heads, head_dim, dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes)
        )
        
    def forward(
        self, 
        imu, 
        audio, 
        video,
        magnetometer=None,
        barometer=None,
        temperature=None,
        spectrometer=None,
        thermal=None
    ):
        """
        Args:
            imu: [batch, seq_len, imu_features]
            audio: [batch, audio_features, audio_seq_len]
            video: [batch, video_seq_len, video_features]
            magnetometer: Optional [batch, seq_len, 3]
            barometer: Optional [batch, seq_len, 1] or [batch, seq_len]
            temperature: Optional [batch, seq_len, 1] or [batch, seq_len]
            spectrometer: Optional [batch, seq_len, num_channels]
            thermal: Optional [batch, seq_len, H, W] or [batch, seq_len, num_sensors]
        Returns:
            Classification logits [batch, num_classes]
        """
        batch_size = imu.shape[0]
        feature_list = []
        
        # Encode core modalities
        imu_features = self.imu_encoder(imu) + self.imu_type_emb
        audio_features = self.audio_encoder(audio) + self.audio_type_emb
        video_features = self.video_encoder(video) + self.video_type_emb
        
        feature_list.extend([imu_features, audio_features, video_features])
        
        # Encode optional modalities if provided
        if self.use_magnetometer and magnetometer is not None:
            mag_features = self.magnetometer_encoder(magnetometer) + self.magnetometer_type_emb
            feature_list.append(mag_features)
        
        if self.use_barometer and barometer is not None:
            baro_features = self.barometer_encoder(barometer) + self.barometer_type_emb
            feature_list.append(baro_features)
        
        if self.use_temperature and temperature is not None:
            temp_features = self.temperature_encoder(temperature) + self.temperature_type_emb
            feature_list.append(temp_features)
        
        if self.use_spectrometer and spectrometer is not None:
            spec_features = self.spectrometer_encoder(spectrometer) + self.spectrometer_type_emb
            feature_list.append(spec_features)
        
        if self.use_thermal and thermal is not None:
            therm_features = self.thermal_encoder(thermal) + self.thermal_type_emb
            feature_list.append(therm_features)
        
        # Concatenate all features into byte array
        byte_array = torch.cat(feature_list, dim=1)
        
        # Initialize latent array (expand for batch)
        latent = self.latent_array.expand(batch_size, -1, -1)
        
        # Process through Perceiver blocks
        for block in self.perceiver_blocks:
            latent = block(latent, byte_array)
        
        # Decode classification via output query
        output_query = self.output_query.expand(batch_size, -1, -1)
        output = output_query + self.output_cross_attn(output_query, latent)
        output = output.squeeze(1)  # [batch, latent_dim]
        
        # Classification
        return self.classifier(output)


# =============================================================================
# Training Script
# =============================================================================

def train_perceiver_io(
    use_magnetometer=False,
    use_barometer=False,
    use_temperature=False,
    use_spectrometer=False,
    use_thermal=False
):
    """
    Main training function for Perceiver IO multimodal classifier.
    
    Args:
        use_magnetometer: Enable magnetometer modality (3-axis magnetic field)
        use_barometer: Enable barometer modality (pressure sensor)
        use_temperature: Enable temperature modality
        use_spectrometer: Enable spectrometer modality (spectral data)
        use_thermal: Enable thermal modality (thermal camera/array)
    """
    
    # Configuration
    base_dir = r"E:\precomputed_data"
    save_path = "perceiver_io_best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 300
    num_latents = 64
    latent_dim = 256
    num_perceiver_blocks = 2
    num_self_attn_per_block = 4
    
    # Optional modality configurations
    # Set to None to disable, or specify input dimensions
    magnetometer_input_size = 3 if use_magnetometer else None  # 3-axis magnetic field
    barometer_input_size = 1 if use_barometer else None        # Single pressure value
    temperature_input_size = 1 if use_temperature else None    # Single temperature value
    spectrometer_channels = 128 if use_spectrometer else None  # Number of spectral bands
    thermal_shape = (32,) if use_thermal else None             # 1D array of 32 thermal sensors
    # For 2D thermal camera, use: thermal_shape = (24, 32)     # 24x32 pixel thermal image
    
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
    
    # Get input shapes from sample batch
    sample = next(iter(train_loader))
    imu_shape = sample["imu"].shape[-1]
    audio_shape = sample["audio"].shape[1:]
    video_shape = sample["video"].shape[1:]
    
    print(f"IMU input size: {imu_shape}")
    print(f"Audio shape: {audio_shape}")
    print(f"Video shape: {video_shape}")
    
    # Print enabled modalities
    print("\nEnabled modalities:")
    print(f"  - IMU (accelerometer + gyroscope): Yes")
    print(f"  - Audio: Yes")
    print(f"  - Video: Yes")
    print(f"  - Magnetometer: {'Yes' if use_magnetometer else 'No'}")
    print(f"  - Barometer: {'Yes' if use_barometer else 'No'}")
    print(f"  - Temperature: {'Yes' if use_temperature else 'No'}")
    print(f"  - Spectrometer: {'Yes' if use_spectrometer else 'No'}")
    print(f"  - Thermal: {'Yes' if use_thermal else 'No'}")
    
    # Initialize model
    model = PerceiverIOFusionClassifier(
        imu_input_size=imu_shape,
        audio_shape=audio_shape,
        video_shape=video_shape,
        num_classes=13,
        latent_dim=latent_dim,
        num_latents=num_latents,
        num_perceiver_blocks=num_perceiver_blocks,
        num_self_attn_per_block=num_self_attn_per_block,
        num_heads=4,
        head_dim=64,
        dropout=0.1,
        # Optional modalities
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
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    
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
            labels = batch["hard_label"].float().to(device)
            
            # Extract optional modalities if available in batch
            magnetometer = batch.get("magnetometer")
            barometer = batch.get("barometer")
            temperature = batch.get("temperature")
            spectrometer = batch.get("spectrometer")
            thermal = batch.get("thermal")
            
            # Move optional modalities to device if present
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
            logits = model(
                imu, audio, video,
                magnetometer=magnetometer,
                barometer=barometer,
                temperature=temperature,
                spectrometer=spectrometer,
                thermal=thermal
            )
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
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
                labels = batch["hard_label"].float().to(device)
                
                # Extract optional modalities if available
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
                
                logits = model(
                    imu, audio, video,
                    magnetometer=magnetometer,
                    barometer=barometer,
                    temperature=temperature,
                    spectrometer=spectrometer,
                    thermal=thermal
                )
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
        
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
                    'imu_input_size': imu_shape,
                    'audio_shape': audio_shape,
                    'video_shape': video_shape,
                    'num_classes': 13,
                    'latent_dim': latent_dim,
                    'num_latents': num_latents,
                    'num_perceiver_blocks': num_perceiver_blocks,
                    'num_self_attn_per_block': num_self_attn_per_block,
                    # Optional modality configs
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


if __name__ == "__main__":
    # Example: Train with core modalities only
    train_perceiver_io()
    
    # Example: Train with all sensor modalities enabled
    # train_perceiver_io(
    #     use_magnetometer=True,
    #     use_barometer=True,
    #     use_temperature=True,
    #     use_spectrometer=True,
    #     use_thermal=True
    # )
