
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import DataLoader
from fixedwindowloader import FixedWindowDataset

from perceiver_io_train import PerceiverIOFusionClassifier


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = PerceiverIOFusionClassifier(
        imu_input_size=config['imu_input_size'],
        audio_shape=config['audio_shape'],
        video_shape=config['video_shape'],
        num_classes=config['num_classes'],
        latent_dim=config['latent_dim'],
        num_latents=config['num_latents'],
        num_perceiver_blocks=config['num_perceiver_blocks'],
        num_self_attn_per_block=config['num_self_attn_per_block'],
        magnetometer_input_size=config.get('magnetometer_input_size'),
        barometer_input_size=config.get('barometer_input_size'),
        temperature_input_size=config.get('temperature_input_size'),
        spectrometer_channels=config.get('spectrometer_channels'),
        thermal_shape=config.get('thermal_shape')
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")
    
    print("Enabled modalities:")
    print(f"  - IMU, Audio, Video: Yes (core)")
    print(f"  - Magnetometer: {'Yes' if config.get('magnetometer_input_size') else 'No'}")
    print(f"  - Barometer: {'Yes' if config.get('barometer_input_size') else 'No'}")
    print(f"  - Temperature: {'Yes' if config.get('temperature_input_size') else 'No'}")
    print(f"  - Spectrometer: {'Yes' if config.get('spectrometer_channels') else 'No'}")
    print(f"  - Thermal: {'Yes' if config.get('thermal_shape') else 'No'}")
    
    return model


def _extract_optional_modalities(batch, device):
    modalities = {}
    for key in ['magnetometer', 'barometer', 'temperature', 'spectrometer', 'thermal']:
        data = batch.get(key)
        if data is not None:
            modalities[key] = data.to(device)
        else:
            modalities[key] = None
    return modalities


def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            imu = batch["imu"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            labels = batch["hard_label"].float().to(device)
            
            opt_mods = _extract_optional_modalities(batch, device)
            
            logits = model(
                imu, audio, video,
                magnetometer=opt_mods['magnetometer'],
                barometer=opt_mods['barometer'],
                temperature=opt_mods['temperature'],
                spectrometer=opt_mods['spectrometer'],
                thermal=opt_mods['thermal']
            )
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    results = {
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_micro': f1_score(all_labels, all_preds, average='micro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'f1_samples': f1_score(all_labels, all_preds, average='samples'),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'accuracy_samples': accuracy_score(all_labels, all_preds),
    }
    
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    results['f1_per_class'] = f1_per_class
    
    return results, all_preds, all_labels, all_probs


def print_evaluation_results(results):
    print("\n" + "="*60)
    print("PERCEIVER IO EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  F1 Score (Macro):    {results['f1_macro']:.4f}")
    print(f"  F1 Score (Micro):    {results['f1_micro']:.4f}")
    print(f"  F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  F1 Score (Samples):  {results['f1_samples']:.4f}")
    print(f"  Precision (Macro):   {results['precision_macro']:.4f}")
    print(f"  Recall (Macro):      {results['recall_macro']:.4f}")
    print(f"  Subset Accuracy:     {results['accuracy_samples']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for i, f1 in enumerate(results['f1_per_class']):
        print(f"  Class {i:2d}: {f1:.4f}")
    
    print("="*60 + "\n")


def find_optimal_threshold(model, val_loader, device):
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            imu = batch["imu"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            labels = batch["hard_label"].float().to(device)
            
            opt_mods = _extract_optional_modalities(batch, device)
            
            logits = model(
                imu, audio, video,
                magnetometer=opt_mods['magnetometer'],
                barometer=opt_mods['barometer'],
                temperature=opt_mods['temperature'],
                spectrometer=opt_mods['spectrometer'],
                thermal=opt_mods['thermal']
            )
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs > threshold).astype(int)
        f1 = f1_score(all_labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold


def main():
    base_dir = r"E:\precomputed_data"
    checkpoint_path = "perceiver_io_best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading test dataset...")
    test_dataset = FixedWindowDataset(
        data_dir=os.path.join(base_dir, "test"),
        imu_window_size=100,
        audio_window_size=250,
        video_window_size=4,
        stride=50
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    val_dataset = FixedWindowDataset(
        data_dir=os.path.join(base_dir, "val"),
        imu_window_size=100,
        audio_window_size=250,
        video_window_size=4,
        stride=50
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    
    print("\nFinding optimal threshold on validation set...")
    optimal_threshold = find_optimal_threshold(model, val_loader, device)
    
    print("\nEvaluating on test set (threshold=0.5)...")
    results_default, _, _, _ = evaluate_model(model, test_loader, device, threshold=0.5)
    print(f"F1 Score (Macro) with threshold 0.5: {results_default['f1_macro']:.4f}")
    
    print(f"\nEvaluating on test set (threshold={optimal_threshold:.2f})...")
    results_optimal, preds, labels, probs = evaluate_model(
        model, test_loader, device, threshold=optimal_threshold
    )
    
    print_evaluation_results(results_optimal)
    
    return results_optimal


if __name__ == "__main__":
    main()
