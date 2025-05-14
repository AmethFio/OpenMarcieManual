from fixedwindowloader import FixedWindowDataset
import os
from torch.utils.data import DataLoader

base_dir = "E:\precomputed_data"

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

test_dataset = FixedWindowDataset(
    data_dir=os.path.join(base_dir, "test"),
    imu_window_size=100,
    audio_window_size=250,
    video_window_size=4,
    stride=50
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

for batch in train_loader:
    print("Train batch:")
    print("  Text:", batch["sentence_embedding"].shape)
    print("  Label:", batch["hard_label"].shape)
    print("  IMU:", batch["imu"].shape)
    print("  Audio:", batch["audio"].shape)
    print("  Video:", batch["video"].shape)
    break
