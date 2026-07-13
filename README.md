# Neural Image Colorization Pipeline
An end-to-end automated image colorization system that maps single-channel grayscale lightness images to two-channel chrominance vectors using deep convolutional networks. Built with PyTorch and deployed via an interactive Gradio web interface.
________________________________________
## Table of Contents
•	Overview
•	Project Type & Domain
•	Dataset
•	Framework & Hardware
•	Engineering Domains
–	Computer Vision & Image Processing
–	Deep Learning & Neural Architecture
–	Data Engineering & Pipeline Optimization
–	Data Validation, Quality Control & Diagnostics
–	Full-Stack Integration & Web UI Deployment
•	Model Architectures
•	Training Configuration
•	Experimental Results
•	How to Run
•	Project Structure
•	Author
________________________________________
## Overview
This project implements a complete pipeline for automatic image colorization using deep learning. The system takes grayscale landscape images as input and predicts realistic color outputs by learning the mapping from lightness (L) to chrominance (a, b) channels in the CIE LAB color space.
Key Achievement: - Best PSNR: 28.26 dB - Best SSIM: 0.3863 - Training Time: 20.43 minutes (50 epochs on 4,000 images) - Model Parameters: 1,861,762 (U-Net with skip connections)
________________________________________
## Project Type & Domain
Attribute	Details
Project Type	Computer Vision / Deep Learning
Primary Task	Image Colorization (Grayscale → RGB)
Color Space	CIE LAB (perceptually decoupled)
Architecture	U-Net with Skip Connections
Framework	PyTorch & Gradio Stack
Hardware	NVIDIA Tesla T4 GPU (CUDA Accelerated)
________________________________________
## Dataset
Primary Dataset — Kaggle Landscape Pictures
Property	Details
Source	Kaggle — arnaud58/landscape-pictures
Size	4,000 high-resolution landscape images
Archive Size	~620 MB
Resolution	Resized to 224 × 224 for training
Split	3,600 training / 400 validation (90/10)
Data Ingestion Method: - Automated Kaggle API authentication and download - Force download and unzip to ephemeral environment storage - On-the-fly BGR → RGB conversion via OpenCV
Supported Image Formats: .jpg, .jpeg, .png (case-insensitive)
________________________________________
## Framework & Hardware
### Software Stack
•	PyTorch — Deep learning framework
•	torchvision — Image transforms and augmentation
•	OpenCV (cv2) — Image I/O and preprocessing
•	scikit-image — LAB color space conversion, PSNR/SSIM metrics
•	Gradio — Interactive web UI deployment
•	NumPy, Pandas, Matplotlib — Data processing and visualization
•	Google Colab — Primary development environment
### Hardware Configuration
•	GPU: NVIDIA Tesla T4 (CUDA Accelerated)
•	Data Transfer: Asynchronous CPU-to-GPU with pin_memory=True
•	Parallel Loading: num_workers=2 for DataLoader optimization
________________________________________
## Engineering Domains
### 1. Computer Vision & Image Processing Domain
Perceptual Color Space Decoupling:
The pipeline transforms input imagery from standard RGB to the CIE LAB color space using skimage.color.rgb2lab:
Channel	Description	Range	Scaled Range
#### L	Luminance (lightness)	0–100	[0.0, 1.0]
#### a	Green-Red spectrum	-128 to +127	[-1.0, 1.0]
#### b	Blue-Yellow spectrum	-128 to +127	[-1.0, 1.0]
Scaling Operations: - L channel: L / 100.0 → [0.0, 1.0] - ab channels: ab / 128.0 → [-1.0, 1.0]
Why LAB over RGB? > Colorizing in the LAB space guarantees that the structural integrity of the original grayscale photo remains perfectly intact, as the network is mathematically restricted from modifying the lightness values.
Evaluation Metrics:
Metric	Formula / Description	Purpose
MSE	Σ(P − T)² / N	Pixel-level error
PSNR	10 × log₁₀(MAX² / MSE)	Signal-to-noise ratio (data_range = 2.0)
SSIM	Luminosity + Contrast + Structure comparison	Perceptual structural coherence
________________________________________
### 2. Deep Learning & Neural Network Architecture Domain
Three Architectures Implemented:
1.	Baseline CNN (Net) — Simple encoder-decoder
2.	Autoencoder (ColorizationAutoencoder) — Bottleneck architecture
3.	U-Net (ColorizationUNet) — Primary trained model with skip connections
#### U-Net Architecture Details:
Input:  [batch, 1, 224, 224]  (L channel)
  ↓
Encoder Path (Contracting):
  enc1:  1 → 64 channels  (2× Conv + ReLU)
  pool1: MaxPool 2×2
  enc2:  64 → 128 channels (2× Conv + ReLU)
  pool2: MaxPool 2×2

#### Bottleneck:
  128 → 256 channels (2× Conv + ReLU)

Decoder Path (Expanding):
  up2:   ConvTranspose 256 → 128
  dec2:  Concat[up2, enc2] → 128 channels (2× Conv + ReLU)
  up1:   ConvTranspose 128 → 64
  dec1:  Concat[up1, enc1] → 64 channels (2× Conv + ReLU)

Output:
  final: 64 → 2 channels (1×1 Conv)
  final_act: Tanh

Output: [batch, 2, 224, 224]  (ab channels)
Total Parameters: 1,861,762
#### Optimization Configuration:
Parameter	Value
Loss Function	Hybrid: MSE + 0.5× L1 + Variance Penalty
Optimizer	Adam
Learning Rate (α)	0.005
Weight Decay	1×10⁻⁴
LR Scheduler	StepLR (step_size=10, gamma=0.5)
Epochs	50
Batch Size	64
Variance Penalty: -torch.std(pred) × 0.05 > Prevents desaturated/gray outputs by penalizing low prediction variance during training.
________________________________________
### 3. Data Engineering & Pipeline Optimization Domain
Automated Data Ingestion: - Kaggle API secure authentication - Force download of 620MB archive - Automatic unzip to working directory
Asynchronous Parallel Loaders:
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,        # Parallel CPU workers
    pin_memory=True,      # Async CPU→GPU transfer
    drop_last=True
)
Custom Dataset Pipeline (ColorizationDataset): 1. Load image via OpenCV (BGR format) 2. Convert to RGB 3. Apply transforms (resize, augmentation) 4. Convert to LAB color space 5. Extract and normalize L and ab channels 6. Return as PyTorch tensors
Strong Chrominance Augmentation:
To combat color convergence stagnation (safe sepia tones), the following augmentation pipeline is deployed:
Transform	Parameters	Purpose
RandomHorizontalFlip	p = 0.5	Spatial diversity
RandomRotation	15°	Orientation diversity
ColorJitter	Brightness=0.2, Contrast=0.2, Saturation=0.3, Hue=0.05	Forces rich color learning
RandomAffine	Translate = 10%	Spatial jitter
The custom saturation jitter forces the network to learn rich, diverse color representations instead of predicting safe sepia tones.
________________________________________
### 4. Data Validation, Quality Control & Diagnostics Domain
Pre-Training Structural Verification:
Before model optimization begins, the data loader pipeline is sanity-checked:
Check	Expected Value	Validation
L tensor shape	[64, 1, 224, 224]	assert verified
ab tensor shape	[64, 2, 224, 224]	assert verified
L value range	[0.0, 1.0]	Range check
ab value range	[-1.0, 1.0]	Range check
Variance Decay Diagnostics:
The training loop continuously monitors the moving standard deviation (Prediction Std) of output tensors:
Condition	Alert	Meaning
σ < 5.0	⚠️ WARNING	Desaturated output — underfitting or mode collapse
σ < 10.0	⚠️ CAUTION	Low variance — consider longer training
σ ≥ 10.0	✓ HEALTHY	Good color variance
Runtime Health Checks: - GPU memory monitoring (torch.cuda.memory_allocated) - Model file size verification - Path consistency validation - Training data color variance audit
________________________________________
### 5. Full-Stack Integration & Web UI Deployment Domain
Interactive Gradio Interface:
Features: - Drag-and-drop image upload panel - Model selection dropdown (CNN / Autoencoder / U-Net) - Real-time colorized output display - Automatic model weight loading (_best.pth, _final.pth)
Robust Serialization & Logging:
Artifact	Format	Content
Model Checkpoints	.pth (PyTorch state_dict)	Best and final weights
Training History	.json	Loss, PSNR, SSIM, LR, time per epoch
Comparison Metrics	.csv	Final model comparison table
Training Curves	.png	Loss, PSNR, SSIM, Prediction Std plots
Checkpoint Strategy: - Best Model: Saved when validation loss reaches new minimum (UNet_best.pth) - Final Model: Saved after last epoch (UNet_final.pth) - Both include: model_state_dict, optimizer_state_dict, epoch, val_loss, psnr, ssim
________________________________________
#### Model Architectures
ColorizationUNet (Primary)
class ColorizationUNet(nn.Module):
    # 1,861,762 parameters
    # Encoder-Decoder with skip connections
    # Input:  [B, 1, 224, 224]  →  Output: [B, 2, 224, 224]
ColorizationAutoencoder
class ColorizationAutoencoder(nn.Module):
    # Bottleneck architecture
    # Encoder: 1→64→128→256 (stride=2 convolutions)
    # Decoder: 256→128→64→2 (transposed convolutions)
Net (Baseline CNN)
class Net(nn.Module):
    # Simple sequential CNN
    # Conv → ReLU → Pool → Conv → ReLU → Pool → Conv → Upsample → Output
________________________________________
Training Configuration
# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
N_TOTAL = 4000          # Total images used
N_VAL = 400             # Validation split (10%)
EPOCHS = 50
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# Loss Components
MSE_WEIGHT = 1.0
L1_WEIGHT = 0.5
VARIANCE_PENALTY_WEIGHT = 0.05

# LR Scheduler
SCHEDULER = StepLR(step_size=10, gamma=0.5)

# Seeds
RANDOM_SEED = 42
________________________________________
Experimental Results
Performance Matrix
Metric	Value
Model Architecture	U-Net (Skip Connections)
Total Parameters	1,861,762
Best Validation Loss	0.0443
Final Validation Loss	0.0443
Best PSNR	28.26 dB
Final PSNR	28.21 dB
Best SSIM	0.3863
Final SSIM	0.3830
Total Training Time	20.43 minutes
Key Success Factors
1.	U-Net Skip Connections: Retain high-frequency spatial details from encoder to decoder
2.	LAB Color Space: Decouples luminance from chrominance, preserving structural integrity
3.	Variance Penalty: Prevents desaturated/gray output convergence
4.	Strong Augmentation: Color jitter forces diverse color learning
5.	Hybrid Loss: MSE + L1 + variance penalty balances pixel accuracy and color diversity
________________________________________
How to Run
Prerequisites
pip install torch torchvision opencv-python scikit-image gradio numpy pandas matplotlib
Option 1: Google Colab (Recommended)
1.	Upload the .ipynb notebook to Google Colab
2.	Upload kaggle.json for API authentication
3.	Run all cells sequentially
4.	The Gradio interface will launch with a public URL
Option 2: Local Environment
# 1. Clone repository
git clone <repo-url>
cd neural-image-colorization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (manual or via Kaggle API)
 Place images in data/train/ and data/val/

# 4. Run training
python train.py

# 5. Launch web UI
python app.py
Notebook Cell Structure
Cell	Purpose
1	Environment Setup & Google Drive Mounting
2	Install & Import Dependencies
3	Dataset Preparation & Verification
4	Custom Dataset Class & Data Loaders
5	Verify Data Loader
6	Model Architectures
7	Evaluation Metrics
8	Unified Training Loop
9	Run Training
10	Comparison & Report
11	Inference & Gradio Interface
12	Diagnostics
13	Debugging & Diagnostics
14	Advanced Troubleshooting
________________________________________
Project Structure
neural-image-colorization/
├── Image_Colorization_Domain_Analysis_Report.pdf   # Full technical report
├── image_colorization.ipynb                        # Main Colab notebook
├── README.md                                       # This file
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── train/                    # 3,600 training images
│   └── val/                      # 400 validation images
│
├── models/
│   ├── UNet_best.pth             # Best checkpoint (min val loss)
│   └── UNet_final.pth            # Final checkpoint (epoch 50)
│
├── results/
│   ├── comparison_metrics.csv    # Model comparison table
│   ├── training_curves.png       # Loss/PSNR/SSIM plots
│   └── UNet_history.json         # Per-epoch training log
│
└── samples/                      # Colorized output samples
________________________________________
# Author
Mahmoona Khan

# Co-Author
Sadia Asif
Komal Yousaf

This project was developed as part of advanced coursework in Computer Vision and Deep Learning, completed in June 2026.
Technologies Used: PyTorch, OpenCV, scikit-image, Gradio, NumPy, Pandas, Matplotlib
Hardware: NVIDIA Tesla T4 GPU via Google Colab
________________________________________
### License
This project is for academic and educational purposes.
