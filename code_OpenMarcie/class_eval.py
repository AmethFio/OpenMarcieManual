import torch
import torch.nn as nn
import os
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset
from classes import LateFusionTransformerClassifier 

base_dir = r"E:\precomputed_data"
save_path = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = FixedWindowDataset(
    data_dir=os.path.join(base_dir, "test"),
    imu_window_size=100,
    audio_window_size=250,
    video_window_size=4,
    stride=50
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

sample = next(iter(test_loader))
imu_shape = sample["imu"].shape[-1]
audio_shape = sample["audio"].shape[1:]
video_shape = sample["video"].shape[1:]

model = LateFusionTransformerClassifier(
    imu_input_size=imu_shape,
    audio_shape=audio_shape,
    video_shape=video_shape,
    num_classes=13
)
model.load_state_dict(torch.load(save_path, map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        imu = batch["imu"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        labels = batch["hard_label"].float().to(device) 
        logits = model(imu, audio, video)
        probs = torch.sigmoid(logits)                   
        preds = (probs > 0.5).int()                    
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

f1 = f1_score(all_labels, all_preds, average='macro') 
print(f"F1 Score on Test Set: {f1:.4f}")
