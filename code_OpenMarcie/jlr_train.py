import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class IMUEncoder(nn.Module):
    def __init__(self, input_dim=3, conv_channels=64, lstm_hidden=128, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * lstm_hidden, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)          
        x = self.conv(x)                
        x = x.permute(0, 2, 1)          
        _, (hn, _) = self.lstm(x)
        h = torch.cat((hn[0], hn[1]), dim=-1)  
        return self.proj(h)

def contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return nn.CrossEntropyLoss()(logits, labels)

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
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

text_encoder = MLPEncoder(768).to(device)
audio_encoder = MLPEncoder(2*250).to(device)
video_encoder = MLPEncoder(4*768).to(device)
imu_encoder = IMUEncoder().to(device)

params = list(text_encoder.parameters()) + \
         list(audio_encoder.parameters()) + \
         list(video_encoder.parameters()) + \
         list(imu_encoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

num_epochs = 300
best_loss = float('inf')
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(num_epochs):
    text_encoder.train()
    audio_encoder.train()
    video_encoder.train()
    imu_encoder.train()

    total_train_loss = 0
    for batch in train_loader:
        text = batch["sentence_embedding"].to(device)
        audio = batch["audio"].flatten(start_dim=1).to(device)
        video = batch["video"].flatten(start_dim=1).to(device)
        imu = batch["imu"].to(device)

        z_text = text_encoder(text)
        z_audio = audio_encoder(audio)
        z_video = video_encoder(video)
        z_imu = imu_encoder(imu)

        loss_ta = contrastive_loss(z_text, z_audio)
        loss_tv = contrastive_loss(z_text, z_video)
        loss_ti = contrastive_loss(z_text, z_imu)
        loss_av = contrastive_loss(z_audio, z_video)
        loss_ai = contrastive_loss(z_audio, z_imu)
        loss_vi = contrastive_loss(z_video, z_imu)

        total_loss = loss_ta + loss_tv + loss_ti + loss_av + loss_ai + loss_vi

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()

    text_encoder.eval()
    audio_encoder.eval()
    video_encoder.eval()
    imu_encoder.eval()

    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            text = batch["sentence_embedding"].to(device)
            audio = batch["audio"].flatten(start_dim=1).to(device)
            video = batch["video"].flatten(start_dim=1).to(device)
            imu = batch["imu"].to(device)

            z_text = text_encoder(text)
            z_audio = audio_encoder(audio)
            z_video = video_encoder(video)
            z_imu = imu_encoder(imu)

            loss_ta = contrastive_loss(z_text, z_audio)
            loss_tv = contrastive_loss(z_text, z_video)
            loss_ti = contrastive_loss(z_text, z_imu)
            loss_av = contrastive_loss(z_audio, z_video)
            loss_ai = contrastive_loss(z_audio, z_imu)
            loss_vi = contrastive_loss(z_video, z_imu)

            val_loss = loss_ta + loss_tv + loss_ti + loss_av + loss_ai + loss_vi
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Avg Train Loss: {total_train_loss / len(train_loader):.4f} | Avg Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': {
                'text': text_encoder.state_dict(),
                'audio': audio_encoder.state_dict(),
                'video': video_encoder.state_dict(),
                'imu': imu_encoder.state_dict()
            },
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss
        }, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch+1} with val loss {best_loss:.4f}")
