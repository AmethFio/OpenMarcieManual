import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset
from torch.utils.data import DataLoader

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

class LateFusionTransformerClassifier(nn.Module):
    def __init__(self, imu_input_size, audio_shape, video_shape, num_classes=5):
        super().__init__()

        self.imu_net = ConvLSTMBlock(imu_input_size, conv_channels=64, lstm_hidden_size=128)
        self.audio_net = MLPEncoder(input_dim=audio_shape[0] * audio_shape[1])
        self.video_net = MLPEncoder(input_dim=video_shape[0] * video_shape[1])

        self.token_proj = nn.Linear(128, 128)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256),
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
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

        return self.classifier(encoded)

base_dir = r"E:\precomputed_data"
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

sample = next(iter(train_loader))
imu_shape = sample["imu"].shape[-1]       
audio_shape = sample["audio"].shape[1:]   
video_shape = sample["video"].shape[1:]  

model = LateFusionTransformerClassifier(
    imu_input_size=imu_shape,
    audio_shape=audio_shape,
    video_shape=video_shape,
    num_classes=13
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
best_val_loss = float('inf')
save_path = "best_model.pth"

for epoch in range(300):
    model.train()
    total_loss = 0
    for batch in train_loader:
        imu = batch["imu"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        labels = batch["hard_label"].float().to(device)

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
            labels = batch["hard_label"].float().to(device)
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

    print(f"Epoch {epoch+1}/10 | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")