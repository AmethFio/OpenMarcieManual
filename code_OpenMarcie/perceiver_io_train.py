
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset


class FourierPositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_bands=64, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.input_dim = input_dim
        self.output_dim = input_dim * (2 * num_bands + 1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        freqs = torch.linspace(1.0, self.max_freq, self.num_bands, device=x.device)
        freqs = freqs.view(1, 1, 1, -1)
        
        x_expanded = x.unsqueeze(-1)
        
        x_freq = x_expanded * freqs * 2 * math.pi
        sin_features = torch.sin(x_freq)
        cos_features = torch.cos(x_freq)
        
        features = torch.cat([x, sin_features.flatten(-2), cos_features.flatten(-2)], dim=-1)
        return features


class CrossAttention(nn.Module):
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
        batch_size = query.shape[0]
        
        query = self.norm_q(query)
        kv = self.norm_kv(kv)
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.out_proj(out)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, head_dim=64, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(dim, dim, num_heads, head_dim, dropout)
        
    def forward(self, x):
        return self.cross_attn(x, x)


class FeedForward(nn.Module):
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
    def __init__(self, latent_dim, input_dim, num_heads=4, head_dim=64, 
                 num_self_attn_layers=6, dropout=0.1):
        super().__init__()
        
        self.cross_attn = CrossAttention(latent_dim, input_dim, num_heads, head_dim, dropout)
        self.cross_ff = FeedForward(latent_dim, dropout=dropout)
        
        self.self_attn_layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(latent_dim, num_heads, head_dim, dropout),
                FeedForward(latent_dim, dropout=dropout)
            ]) for _ in range(num_self_attn_layers)
        ])
        
    def forward(self, latent, inputs):
        latent = latent + self.cross_attn(latent, inputs)
        latent = latent + self.cross_ff(latent)
        
        for self_attn, ff in self.self_attn_layers:
            latent = latent + self_attn(latent)
            latent = latent + ff(latent)
        
        return latent


class IMUEncoder(nn.Module):
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
    def __init__(self, num_channels, hidden_dim=128, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.temporal_proj = nn.Linear(num_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        batch_size, seq_len, num_channels = x.shape
        
        x = self.temporal_proj(x)
        x, _ = self.lstm(x)
        x = self.proj(x)
        return self.norm(x)


class ThermalEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim=128, output_dim=256):
        super().__init__()
        self.input_shape = input_shape
        
        if len(input_shape) == 2:
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
        else:
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


class PerceiverIOFusionClassifier(nn.Module):
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
        
        self.use_magnetometer = magnetometer_input_size is not None
        self.use_barometer = barometer_input_size is not None
        self.use_temperature = temperature_input_size is not None
        self.use_spectrometer = spectrometer_channels is not None
        self.use_thermal = thermal_shape is not None
        
        self.imu_encoder = IMUEncoder(imu_input_size, output_dim=latent_dim)
        self.audio_encoder = AudioEncoder(audio_shape[0], output_dim=latent_dim)
        self.video_encoder = VideoEncoder(video_shape[1], output_dim=latent_dim)
        
        self.imu_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.audio_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.video_type_emb = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
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
        
        self.latent_array = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)
        
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
        
        self.output_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        self.output_cross_attn = CrossAttention(
            latent_dim, latent_dim, num_heads, head_dim, dropout
        )
        
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
        batch_size = imu.shape[0]
        feature_list = []
        
        imu_features = self.imu_encoder(imu) + self.imu_type_emb
        audio_features = self.audio_encoder(audio) + self.audio_type_emb
        video_features = self.video_encoder(video) + self.video_type_emb
        
        feature_list.extend([imu_features, audio_features, video_features])
        
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
        
        byte_array = torch.cat(feature_list, dim=1)
        
        latent = self.latent_array.expand(batch_size, -1, -1)
        
        for block in self.perceiver_blocks:
            latent = block(latent, byte_array)
        
        output_query = self.output_query.expand(batch_size, -1, -1)
        output = output_query + self.output_cross_attn(output_query, latent)
        output = output.squeeze(1)
        
        return self.classifier(output)


def train_perceiver_io(
    use_magnetometer=False,
    use_barometer=False,
    use_temperature=False,
    use_spectrometer=False,
    use_thermal=False
):
    
    base_dir = r"E:\precomputed_data"
    save_path = "perceiver_io_best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 300
    num_latents = 64
    latent_dim = 256
    num_perceiver_blocks = 2
    num_self_attn_per_block = 4
    
    magnetometer_input_size = 3 if use_magnetometer else None
    barometer_input_size = 1 if use_barometer else None
    temperature_input_size = 1 if use_temperature else None
    spectrometer_channels = 128 if use_spectrometer else None
    thermal_shape = (32,) if use_thermal else None
    
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
    
    sample = next(iter(train_loader))
    imu_shape = sample["imu"].shape[-1]
    audio_shape = sample["audio"].shape[1:]
    video_shape = sample["video"].shape[1:]
    
    print(f"IMU input size: {imu_shape}")
    print(f"Audio shape: {audio_shape}")
    print(f"Video shape: {video_shape}")
    
    print("\nEnabled modalities:")
    print(f"  - IMU (accelerometer + gyroscope): Yes")
    print(f"  - Audio: Yes")
    print(f"  - Video: Yes")
    print(f"  - Magnetometer: {'Yes' if use_magnetometer else 'No'}")
    print(f"  - Barometer: {'Yes' if use_barometer else 'No'}")
    print(f"  - Temperature: {'Yes' if use_temperature else 'No'}")
    print(f"  - Spectrometer: {'Yes' if use_spectrometer else 'No'}")
    print(f"  - Thermal: {'Yes' if use_thermal else 'No'}")
    
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
        magnetometer_input_size=magnetometer_input_size,
        barometer_input_size=barometer_input_size,
        temperature_input_size=temperature_input_size,
        spectrometer_channels=spectrometer_channels,
        thermal_shape=thermal_shape
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            imu = batch["imu"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            labels = batch["hard_label"].float().to(device)
            
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imu = batch["imu"].to(device)
                audio = batch["audio"].to(device)
                video = batch["video"].to(device)
                labels = batch["hard_label"].float().to(device)
                
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
    train_perceiver_io()
    
