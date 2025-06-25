# CDFN

The relevant code for the paper "CDFN: Credibility-Driven Fusion Network for Multimodal Sentiment Analysis".

## Requirements

- Python >= 3.9
- PyTorch ==2.2.1+cu118
- Specific environmental requirements can be found in the file "requirements. txt"

## Train
Operation process:

- 1: Program entrance: folder 'train.py'
 
	
- 2: Hyperparameter optimization entrance: 'train_optuna.py'


## Dataset
  Obtain MOSI and MOSEI datasets from official channels and put them into the datasets file.
  
  Specific operation tips can be found in the datasets file.
  
Download reference link: https://github.com/declare-lab/MISA

Data Download

Install CMU Multimodal SDK. Ensure, you can perform from mmsdk import mmdatasdk.

Option 1: Download pre-computed splits and place the contents inside datasets folder.

Option 2: Re-create splits by downloading data from MMSDK. For this, simply run the code as detailed next.
