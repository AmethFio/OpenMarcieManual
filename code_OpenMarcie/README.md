
# Project Setup Instructions

```bash
# Create the environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate myenv

# Install any additional pip requirements
pip install -r requirements.txt

# Run your Precompute Python script
python loader.py

# Run your  fixed_window compute script
python fixedwindowloader.py

# Run your  training script
python '*'_train.py

# Run your  eval script
python '*'_eval.py

```
</div>


## Results

### Table 1: Human Activity Recognition

**Macro F1 scores** for Scenario (a): Bicycle Assembly and Scenario (b): 3D Printer Assembly, with and without the null class.

| **Modality**                 | **Scenario (a)** <br> Without Null | With Null         | **Scenario (b)** <br> Without Null | With Null         |
| ---------------------------- | ---------------------------------- | ----------------- | ---------------------------------- | ----------------- |
| Inertial                     | 0.834 ± 0.007                      | 0.811 ± 0.007     | 0.750 ± 0.015                      | 0.674 ± 0.003     |
| Acoustic                     | 0.489 ± 0.018                      | 0.469 ± 0.017     | 0.425 ± 0.004                      | 0.432 ± 0.005     |
| Vision                       | 0.757 ± 0.011                      | 0.729 ± 0.011     | 0.705 ± 0.004                      | 0.655 ± 0.003     |
| Inertial + Acoustic          | 0.803 ± 0.012                      | 0.782 ± 0.010     | 0.744 ± 0.004                      | 0.666 ± 0.003     |
| Acoustic + Vision            | 0.739 ± 0.016                      | 0.714 ± 0.013     | 0.695 ± 0.003                      | 0.646 ± 0.003     |
| **Inertial + Vision**        | **0.882 ± 0.009**                  | **0.851 ± 0.009** | **0.773 ± 0.000**                  | **0.685 ± 0.000** |
| Inertial + Acoustic + Vision | 0.859 ± 0.010                      | 0.831 ± 0.011     | 0.763 ± 0.003                      | 0.676 ± 0.003     |

---

### Table 2: Open Vocabulary Captioning

**Cosine similarity scores** for Scenario (a): Bicycle Assembly and Scenario (b): 3D Printer Assembly.

| **Modality**                 | **Scenario (a)** <br> Without Null | With Null         | **Scenario (b)** <br> Without Null | With Null         |
| ---------------------------- | ---------------------------------- | ----------------- | ---------------------------------- | ----------------- |
| Inertial                     | 0.518 ± 0.023                      | 0.501 ± 0.022     | 0.642 ± 0.002                      | 0.640 ± 0.002     |
| Acoustic                     | 0.361 ± 0.030                      | 0.341 ± 0.018     | 0.316 ± 0.003                      | 0.323 ± 0.004     |
| Vision                       | 0.479 ± 0.016                      | 0.463 ± 0.014     | 0.632 ± 0.002                      | 0.631 ± 0.003     |
| Inertial + Acoustic          | 0.512 ± 0.021                      | 0.493 ± 0.020     | 0.644 ± 0.002                      | 0.641 ± 0.003     |
| Acoustic + Vision            | 0.466 ± 0.025                      | 0.444 ± 0.012     | 0.626 ± 0.003                      | 0.625 ± 0.004     |
| **Inertial + Vision**        | **0.561 ± 0.016**                  | **0.531 ± 0.014** | **0.655 ± 0.000**                  | **0.655 ± 0.000** |
| Inertial + Acoustic + Vision | 0.547 ± 0.020                      | 0.519 ± 0.017     | 0.647 ± 0.001                      | 0.646 ± 0.003     |

---

### Table 3: Cross-Modal Alignment

**Recall\@1, Recall\@5, and Top-1 metrics** for Scenario (a): Bicycle Assembly and Scenario (b): 3D Printer Assembly.

| **Modality**                        | **Recall\@1 (a)** | **Recall\@5 (a)** | **Top-1 (a)**     | **Recall\@1 (b)** | **Recall\@5 (b)** | **Top-1 (b)**     |
| ----------------------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Inertial + Text                     | 0.324 ± 0.016     | 0.655 ± 0.025     | 0.481 ± 0.018     | 0.312 ± 0.016     | 0.642 ± 0.026     | 0.468 ± 0.019     |
| Acoustic + Text                     | 0.241 ± 0.014     | 0.583 ± 0.025     | 0.342 ± 0.016     | 0.227 ± 0.013     | 0.567 ± 0.022     | 0.329 ± 0.015     |
| Vision + Text                       | 0.437 ± 0.015     | 0.768 ± 0.017     | 0.556 ± 0.016     | 0.421 ± 0.013     | 0.751 ± 0.018     | 0.541 ± 0.014     |
| Inertial + Acoustic + Text          | 0.347 ± 0.014     | 0.679 ± 0.019     | 0.495 ± 0.017     | 0.334 ± 0.015     | 0.663 ± 0.017     | 0.479 ± 0.018     |
| Acoustic + Vision + Text            | 0.412 ± 0.013     | 0.740 ± 0.020     | 0.533 ± 0.015     | 0.395 ± 0.014     | 0.723 ± 0.019     | 0.517 ± 0.014     |
| **Inertial + Vision + Text**        | **0.485 ± 0.014** | **0.803 ± 0.019** | **0.587 ± 0.016** | **0.467 ± 0.013** | **0.787 ± 0.015** | **0.570 ± 0.016** |
| Inertial + Acoustic + Vision + Text | 0.470 ± 0.015     | 0.795 ± 0.019     | 0.579 ± 0.016     | 0.453 ± 0.014     | 0.779 ± 0.018     | 0.563 ± 0.016     |

