from loaderbike import EgoChestMultiSessionDataset
import os
import torch
from tqdm import tqdm

base_path = r"E:\BikeExperiment"
csv_hard_list = []
csv_soft_list = []
imu_csv_list = []
audio_file_list = []
video_file_list = []

for i in range(11, 12):
    vol_path = os.path.join(base_path, f"Vol{i}", "ProcessedData")
    csv_hard_list.append(os.path.join(vol_path, "Labels", f"processed_vol{i}_EgoChestVideoLabelsSec_Numbered_Formatted.csv"))
    csv_soft_list.append(os.path.join(vol_path, "Labels", f"processed_vol{i}_EgoChestVideoLabelsSec_SoftLabels_Rich.csv"))
    imu_csv_list.append(os.path.join(vol_path, "WearableData", "ImuAndBaro", "RightWristLabelled.csv"))
    audio_file_list.append(os.path.join(vol_path, "WearableData", "Sound", "sound_res_right_instrumental.wav"))
    video_file_list.append(os.path.join(vol_path, "WearableData", "EgoVideoChest", f"BikeExpVol{i}_RGB_512.mp4"))

dataset = EgoChestMultiSessionDataset(csv_hard_list, csv_soft_list, imu_csv_list, audio_file_list, video_file_list)

save_dir = r"precomputed_data\val"
os.makedirs(save_dir, exist_ok=True)

for i, sample in enumerate(tqdm(dataset)):
    torch.save(sample, os.path.join(save_dir, f"sample_{i}.pt"))
