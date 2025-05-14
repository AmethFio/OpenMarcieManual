import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset
from regress import LateFusionTransformerRegressor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

test_data_dir = r"E:\precomputed_data\test"
batch_size = 4
model_path = "best_regressor_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = FixedWindowDataset(
    data_dir=test_data_dir,
    imu_window_size=100,
    audio_window_size=250,
    video_window_size=4,
    stride=50
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

sample = next(iter(test_loader))
imu_input_size = sample["imu"].shape[-1]
audio_shape = sample["audio"].shape[1:]
video_shape = sample["video"].shape[1:]
output_dim = sample["sentence_embedding"].shape[-1]

model = LateFusionTransformerRegressor(
    imu_input_size=imu_input_size,
    audio_shape=audio_shape,
    video_shape=video_shape,
    output_dim=output_dim
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_cos_sims = []

with torch.no_grad():
    for batch in test_loader:
        imu = batch["imu"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        targets = batch["sentence_embedding"].float().to(device)

        outputs = model(imu, audio, video)
        outputs_norm = F.normalize(outputs, p=2, dim=1)
        targets_norm = F.normalize(targets, p=2, dim=1)

        cos_sims = torch.sum(outputs_norm * targets_norm, dim=1).cpu().numpy()
        all_cos_sims.extend(cos_sims)

mean_cos_sim = np.mean(all_cos_sims)
std_cos_sim = np.std(all_cos_sims)
print(f"Test Cosine Similarity: Mean = {mean_cos_sim:.4f}, Std = {std_cos_sim:.4f}")
