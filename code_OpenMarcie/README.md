
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
