import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

class FixedWindowDataset(Dataset):
    def __init__(
        self,
        data_dir,
        imu_window_size=100,
        audio_window_size=250,
        video_window_size=8,
        stride=50
    ):
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        ])
        self.imu_window_size = imu_window_size
        self.audio_window_size = audio_window_size
        self.video_window_size = video_window_size
        self.stride = stride

        self.windowed_samples = []
        self._build_index()

    def _build_index(self):
        for file_path in tqdm(self.file_paths, desc="Indexing windowed segments"):
            sample = torch.load(file_path)

            sentence = sample["sentence_embedding"]
            label = sample["hard_label"]
            imu = sample["imu_segment"]
            audio = sample["audio_embedding"].squeeze(0) 
            video = sample["video_embedding"]            

            imu_len = imu.shape[0]
            audio_len = audio.shape[1]
            video_len = video.shape[0]

            max_win = min(
                (imu_len - self.imu_window_size) // self.stride + 1,
                (audio_len - self.audio_window_size) // self.stride + 1,
                (video_len - self.video_window_size) // self.stride + 1
            )

            for i in range(max_win):
                start = i * self.stride
                imu_win = imu[start:start + self.imu_window_size]
                audio_win = audio[:, start:start + self.audio_window_size]
                video_win = video[start:start + self.video_window_size]

                self.windowed_samples.append({
                    "sentence_embedding": sentence,
                    "hard_label": label,
                    "imu": imu_win,
                    "audio": audio_win,
                    "video": video_win
                })

    def __len__(self):
        return len(self.windowed_samples)

    def __getitem__(self, idx):
        item = self.windowed_samples[idx]
        return {
            "sentence_embedding": torch.tensor(item["sentence_embedding"], dtype=torch.float),
            "hard_label": torch.tensor(item["hard_label"], dtype=torch.float),
            "imu": torch.tensor(item["imu"], dtype=torch.float),
            "audio": torch.tensor(item["audio"], dtype=torch.float),
            "video": torch.tensor(item["video"], dtype=torch.float),
        }

dataset = FixedWindowDataset(
    data_dir="precomputed_data",
    imu_window_size=100,
    audio_window_size=250,
    video_window_size=4,
    stride=50
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    print("Text:", batch["sentence_embedding"].shape)   
    print("Label:", batch["hard_label"].shape)          
    print("IMU:", batch["imu"].shape)                   
    print("Audio:", batch["audio"].shape)               
    print("Video:", batch["video"].shape)               
    break