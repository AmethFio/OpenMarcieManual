import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import decord
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import EncodecModel, AutoProcessor, ViTModel, ViTFeatureExtractor
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

decord.bridge.set_bridge("torch")

class EgoChestMultiSessionDataset(Dataset):
    def __init__(self, hard_csvs, soft_csvs, imu_csvs, audio_files, video_files, model_name='sentence-transformers/gtr-t5-base'):
        assert len(hard_csvs) == len(soft_csvs) == len(imu_csvs) == len(audio_files) == len(video_files)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        self.encodec_model = EncodecModel.from_pretrained("facebook/encodec_16khz").to(self.device)
        self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_16khz")
        self.target_sr = self.audio_processor.sampling_rate

        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)
        self.vit_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vision_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=self.vit_processor.image_mean, std=self.vit_processor.image_std)
        ])

        # Caches
        self.sentence_cache = {}
        self.audio_cache = {}
        self.video_cache = {}

        all_sentences = []
        all_labels_raw = []
        all_imu_segments = []
        all_audio_embeddings = []
        all_video_embeddings = []

        for h_csv, s_csv, imu_csv, audio_path, video_path in zip(hard_csvs, soft_csvs, imu_csvs, audio_files, video_files):
            print(f"Processing: {h_csv}, {audio_path}, {video_path}")
            assert os.path.exists(h_csv) and os.path.exists(s_csv) and os.path.exists(imu_csv) and os.path.exists(audio_path) and os.path.exists(video_path)

            hard_df = pd.read_csv(h_csv)
            soft_df = pd.read_csv(s_csv)
            imu_df = pd.read_csv(imu_csv, low_memory=False)
            merged_df = pd.merge(hard_df, soft_df, on=["Start_Time", "End_Time"])
            merged_df = merged_df.dropna(subset=["Sentence"]).reset_index(drop=True)

            imu_df = imu_df[["Time", "Lax", "Lay", "Laz"]].dropna()
            imu_time = imu_df["Time"].values
            imu_acc = imu_df[["Lax", "Lay", "Laz"]].values

            waveform, original_sr = torchaudio.load(audio_path)
            if original_sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            vr = decord.VideoReader(video_path)
            video_fps = vr.get_avg_fps()

            def parse_labels(label_str):
                if pd.isna(label_str):
                    return ["none"]
                labels = [lbl.strip() for lbl in str(label_str).strip('"').split(',') if lbl.strip()]
                return labels if labels else ["none"]

            merged_df["Label_List"] = merged_df["Label_Numbers"].apply(parse_labels)

            for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc=f"Processing segments for {os.path.basename(video_path)}"):
                start, end = row["Start_Time"], row["End_Time"]

                mask = (imu_time >= start) & (imu_time <= end)
                imu_segment = imu_acc[mask]
                if imu_segment.shape[0] == 0:
                    imu_segment = np.zeros((1, 3))

                audio_key = (start, end)
                if audio_key in self.audio_cache:
                    audio_embedding = self.audio_cache[audio_key]
                else:
                    if start == end:
                        audio_segment = torch.zeros((1, int(self.target_sr * 0.01)))
                    else:
                        start_sample = int(start * self.target_sr)
                        end_sample = int(end * self.target_sr)
                        audio_segment = waveform[:, start_sample:end_sample]
                        if audio_segment.shape[1] == 0:
                            audio_segment = torch.zeros((1, int(self.target_sr * 0.01)))
                    if audio_segment.shape[0] > 1:
                        audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)
                    mono_audio = audio_segment.squeeze(0)
                    with torch.no_grad():
                        audio_inputs = self.audio_processor(raw_audio=mono_audio, sampling_rate=self.target_sr, return_tensors="pt").to(self.device)
                        encoder_outputs = self.encodec_model.encode(audio_inputs["input_values"], audio_inputs["padding_mask"])
                        audio_embedding = encoder_outputs.audio_codes.squeeze(0).cpu()
                    self.audio_cache[audio_key] = audio_embedding

                video_key = (start, end)
                if video_key in self.video_cache:
                    video_embedding = self.video_cache[video_key]
                else:
                    start_idx = int(start * video_fps)
                    end_idx = int(end * video_fps)

                    if end_idx <= start_idx:
                        video_embedding = torch.zeros((1, self.vit_model.config.hidden_size))
                    else:
                        # Sample frames at 8 FPS regardless of segment duration
                        target_fps = 8
                        total_frames = int((end - start) * target_fps)
                        frame_times = np.linspace(start, end, num=total_frames, endpoint=False)
                        frame_indices = (frame_times * video_fps).astype(int)
                        frame_indices = np.clip(frame_indices, 0, len(vr) - 1)

                        frame_indices = np.clip(frame_indices, 0, len(vr) - 1)
                        frames_raw = vr.get_batch(frame_indices).permute(0, 3, 1, 2)
                        transformed_frames = torch.stack([
                            self.vision_transform(to_pil_image(f.cpu())) for f in frames_raw
                        ])
                        video_embeddings = []
                        transformed_frames = transformed_frames.to(self.device)
                        batch_size = 4
                        for i in range(0, transformed_frames.size(0), batch_size):
                            batch = transformed_frames[i:i + batch_size]
                            with torch.no_grad():
                                outputs = self.vit_model(pixel_values=batch)
                                cls_tokens = outputs.last_hidden_state[:, 0]
                                video_embeddings.append(cls_tokens.cpu())
                        video_embedding = torch.cat(video_embeddings, dim=0)
                        self.video_cache[video_key] = video_embedding

                sentence_text = row["Sentence"]
                if sentence_text in self.sentence_cache:
                    sentence_embedding = self.sentence_cache[sentence_text]
                else:
                    sentence_embedding = self.embedding_model.encode([sentence_text], convert_to_numpy=True)[0]
                    self.sentence_cache[sentence_text] = sentence_embedding

                all_sentences.append(sentence_text)
                all_labels_raw.append(row["Label_List"])
                all_imu_segments.append(imu_segment)
                all_audio_embeddings.append(audio_embedding)
                all_video_embeddings.append(video_embedding)

        label_classes = ["none"] + [str(i) for i in range(1, 12)]
        self.mlb = MultiLabelBinarizer(classes=label_classes)
        multi_hot_labels = self.mlb.fit_transform(all_labels_raw)

        self.data = []
        for sent, label, imu, audio_emb, video_emb in zip(all_sentences, multi_hot_labels, all_imu_segments, all_audio_embeddings, all_video_embeddings):
            self.data.append({
                "sentence_embedding": self.sentence_cache[sent],
                "hard_label": label,
                "imu_segment": imu,
                "audio_embedding": audio_emb,
                "video_embedding": video_emb
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "sentence_embedding": torch.tensor(item["sentence_embedding"], dtype=torch.float).to(self.device),
            "hard_label": torch.tensor(item["hard_label"], dtype=torch.float).to(self.device),
            "imu_segment": torch.tensor(item["imu_segment"], dtype=torch.float).to(self.device),
            "audio_embedding": item["audio_embedding"],
            "video_embedding": item["video_embedding"]
        }

def variable_length_collate(batch):
    return batch

if __name__ == "__main__":
    
    csv_hard_list = []
    csv_soft_list = []
    imu_csv_list = []
    audio_file_list = []
    video_file_list = []

    base_path = r"E:\BikeExperiment"
    for i in range(1, 11):
        vol_path = os.path.join(base_path, f"Vol{i}", "ProcessedData")
        csv_hard_list.append(os.path.join(vol_path, "Labels", f"processed_vol{i}_EgoChestVideoLabelsSec_Numbered_Formatted.csv"))
        csv_soft_list.append(os.path.join(vol_path, "Labels", f"processed_vol{i}_EgoChestVideoLabelsSec_SoftLabels_Rich.csv"))
        imu_csv_list.append(os.path.join(vol_path, "WearableData", "ImuAndBaro", "RightWristLabelled.csv"))
        audio_file_list.append(os.path.join(vol_path, "WearableData", "Sound", "sound_res_right_instrumental.wav"))
        video_file_list.append(os.path.join(vol_path, "WearableData", "EgoVideoChest", f"BikeExpVol{i}_RGB_512.mp4"))

    dataset = EgoChestMultiSessionDataset(csv_hard_list, csv_soft_list, imu_csv_list, audio_file_list, video_file_list)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=variable_length_collate)

    for batch in dataloader:
        for sample in batch:
            print("Text:", sample["sentence_embedding"])
            print("Label:", sample["hard_label"])
            print("IMU:", sample["imu_segment"])
            print("Audio:", sample["audio_embedding"])
            print("Video:", sample["video_embedding"])