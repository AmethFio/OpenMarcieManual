import torch
import torch.nn as nn
import os
import torch
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset

class ConvLSTMBlock(nn.Module):
    def __init__(self, input_size, conv_channels, lstm_hidden_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden_size, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return x[:, -1, :]

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)

class LateFusionTransformerRegressor(nn.Module):
    def __init__(self, imu_input_size, audio_shape, video_shape, output_dim):
        super().__init__()
        self.imu_net = ConvLSTMBlock(imu_input_size, conv_channels=64, lstm_hidden_size=128)
        self.audio_net = MLPEncoder(input_dim=audio_shape[0] * audio_shape[1])
        self.video_net = MLPEncoder(input_dim=video_shape[0] * video_shape[1])

        self.token_proj = nn.Linear(128, 128)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256),
            num_layers=2
        )

        self.regressor = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, imu, audio, video):
        imu_feat = self.imu_net(imu)
        audio_feat = self.audio_net(audio)
        video_feat = self.video_net(video)

        fused = torch.stack([imu_feat, audio_feat, video_feat], dim=1)
        fused = self.token_proj(fused)

        fused = fused.permute(1, 0, 2)
        encoded = self.transformer(fused)
        encoded = encoded.permute(1, 0, 2).flatten(1)

        return self.regressor(encoded)

base_dir = r"E:\precomputed_data"
train_dataset = FixedWindowDataset(os.path.join(base_dir, "train"), 100, 250, 4, 50)
val_dataset = FixedWindowDataset(os.path.join(base_dir, "val"), 100, 250, 4, 50)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

sample = next(iter(train_loader))
imu_shape = sample["imu"].shape[-1]
audio_shape = sample["audio"].shape[1:]
video_shape = sample["video"].shape[1:]
sentence_shape = sample["sentence_embedding"].shape[-1]

model = LateFusionTransformerRegressor(
    imu_input_size=imu_shape,
    audio_shape=audio_shape,
    video_shape=video_shape,
    output_dim=sentence_shape
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
best_val_loss = float('inf')
save_path = "best_regressor_model.pth"

for epoch in range(300):
    model.train()
    total_loss = 0
    for batch in train_loader:
        imu = batch["imu"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        labels = batch["sentence_embedding"].float().to(device)

        logits = model(imu, audio, video)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            imu = batch["imu"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            labels = batch["sentence_embedding"].float().to(device)

            logits = model(imu, audio, video)
            val_loss = criterion(logits, labels)
            total_val_loss += val_loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1}")

    print(f"Epoch {epoch+1}/300 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")