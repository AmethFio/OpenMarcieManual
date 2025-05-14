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

def compute_recall_at_k(query, candidates, k):
    query = F.normalize(query, dim=1)
    candidates = F.normalize(candidates, dim=1)
    sims = torch.matmul(query, candidates.T)
    topk = sims.topk(k, dim=1).indices
    target = torch.arange(query.size(0)).to(query.device).unsqueeze(1)
    correct = (topk == target).any(dim=1).float()
    return correct.mean().item()

def evaluate_recall(text_encoder, audio_encoder, video_encoder, imu_encoder, dataloader):
    text_encoder.eval()
    audio_encoder.eval()
    video_encoder.eval()
    imu_encoder.eval()

    all_text, all_audio, all_video, all_imu = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            text = batch["sentence_embedding"].to(device)
            audio = batch["audio"].flatten(start_dim=1).to(device)
            video = batch["video"].flatten(start_dim=1).to(device)
            imu = batch["imu"].to(device)

            all_text.append(text_encoder(text))
            all_audio.append(audio_encoder(audio))
            all_video.append(video_encoder(video))
            all_imu.append(imu_encoder(imu))

    z_text = torch.cat(all_text, dim=0)
    z_audio = torch.cat(all_audio, dim=0)
    z_video = torch.cat(all_video, dim=0)
    z_imu = torch.cat(all_imu, dim=0)

    print("Retrieval Evaluation (shared embedding space)")
    print("-" * 50)
    print("Pair                 | Recall@1 | Recall@5")
    print("---------------------|-----------|-----------")
    print(f"Text → Audio         | {compute_recall_at_k(z_text, z_audio, 1):.4f}     | {compute_recall_at_k(z_text, z_audio, 5):.4f}")
    print(f"Text → Video         | {compute_recall_at_k(z_text, z_video, 1):.4f}     | {compute_recall_at_k(z_text, z_video, 5):.4f}")
    print(f"Text → IMU           | {compute_recall_at_k(z_text, z_imu, 1):.4f}     | {compute_recall_at_k(z_text, z_imu, 5):.4f}")
    print(f"Audio → Video        | {compute_recall_at_k(z_audio, z_video, 1):.4f}     | {compute_recall_at_k(z_audio, z_video, 5):.4f}")
    print(f"Audio → IMU          | {compute_recall_at_k(z_audio, z_imu, 1):.4f}     | {compute_recall_at_k(z_audio, z_imu, 5):.4f}")
    print(f"Video → IMU          | {compute_recall_at_k(z_video, z_imu, 1):.4f}     | {compute_recall_at_k(z_video, z_imu, 5):.4f}")

base_dir = r"E:\precomputed_data"
test_dataset = FixedWindowDataset(
    data_dir=os.path.join(base_dir, "test"),
    imu_window_size=100,
    audio_window_size=250,
    video_window_size=4,
    stride=50
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

checkpoint_path = os.path.join("checkpoints", "best_model.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)

text_encoder = MLPEncoder(768).to(device)
audio_encoder = MLPEncoder(2*250).to(device)
video_encoder = MLPEncoder(4*768).to(device)
imu_encoder = IMUEncoder().to(device)

text_encoder.load_state_dict(checkpoint['model_state_dict']['text'])
audio_encoder.load_state_dict(checkpoint['model_state_dict']['audio'])
video_encoder.load_state_dict(checkpoint['model_state_dict']['video'])
imu_encoder.load_state_dict(checkpoint['model_state_dict']['imu'])

evaluate_recall(text_encoder, audio_encoder, video_encoder, imu_encoder, test_loader)
