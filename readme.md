## Environment Configuration

### 1. Training Environment
- **Image**: PyTorch 2.3.0 Python 3.12 (ubuntu22.04) CUDA 12.1
- **GPU**: RTX 4090 (24GB) × 1
- **CPU**: 16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- **Dependencies**: See `train_requirements.txt`


### 2. Testing Environment
- **Image**: PyTorch 2.3.1 Python 3.8.19
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0GB)
- **CPU**: 20 vCPU 12th Gen Intel(R) Core(TM) i7-1270H
- **Dependencies**: See `test_requirements.txt`


## Reproducibility Guarantee

### 1. Random Seed Initialization
To ensure reproducible experimental results, all random seeds are fixed throughout training and testing:
```python
import os
import random
import numpy as np
import torch

# Set random seeds
os.environ['PYTHONHASHSEED'] = '2025'
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
## Test Set Details
### 1. Test Set Integrity Verification
The SHA-256 hash value of the test set compressed file is:  
`357d917dc5829c9bac64a5ba035df27da4974dcd787014e429ade1c39392a6d2`  

### 2. Test Set Label Generation
Run the `create_label.py` script to generate the test set label file, and place the generated label file in the corresponding directory of the test code before conducting the test.

### 3. Test Set Description
This model is primarily designed for Deepfake detection on Asian faces, so the AI-generated faces in the test set are all Asian faces. The real face samples are selected from the CelebA dataset, and these samples do not appear in the training set, which ensures the objectivity and validity of the model evaluation.

## Impact of Random Seed Fixing
To evaluate the stability and reproducibility of our model, we conducted experiments with and without fixed random seeds:

- **Without Fixed Random Seeds**: The model weights are saved as `improved_model_random.pth`, achieving an accuracy of **91%** on the test set. While this result is higher, it may not be reproducible due to the inherent randomness in the training process (e.g., weight initialization, data shuffling).

- **With Fixed Random Seeds**: The model weights are saved as `improved_model.pth`, achieving an accuracy of **82%** on the test set. Although the accuracy is slightly lower, this configuration ensures that the experimental results can be stably reproduced across different runs and environments, which is crucial for scientific research.