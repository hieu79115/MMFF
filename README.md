# PCINet: Pose-Guided Context Refinement and Global Pose-Context Interaction for Action Recognition

> **Note on Repository Naming:** *`MMFF` (Multimodal Feature Fusion Framework) is the official reference implementation codebase for **PCINet** ("PCINet: Pose-Guided Context Refinement and Global Pose-Context Interaction for Action Recognition").*

## 📌 Method Overview

**PCINet** is a multimodal action recognition framework designed to address the challenge of distinguishing actions with similar human body configurations by effectively coordinating pose-guided spatial visual refinement with subsequent global multimodal interaction.

The framework consists of two core modules:

1. **Context Feature Enhancement Block (CFEB)**: Uses temporally aggregated joint-level pose representations from an ST-GCN backbone to guide the spatial refinement of Xception-extracted RGB feature maps via cross-modal attention before global pooling.

2. **Context-Pose Interaction Block (CPIB)**: Projects enhanced contextual and pose representations into compact modality tokens and models their global inter-modality dependencies using a multi-head Transformer encoder.

## 📊 Benchmark Summary

PCINet achieves state-of-the-art performance across key benchmark datasets:

| Dataset | Modality | Setting / Protocol | Accuracy (%) | F1-Score (%) | 
| ----- | ----- | ----- | ----- | ----- | 
| **UTD-MHAD** | Skeleton + RGB | Subject Split (Odd/Even) | **96.28%** | — | 
| **NW-UCLA** | Skeleton + RGB | Cross-View (V1,V2 / V3) | **93.49%** | — | 
| **HCMUE-SB** | Skeleton + RGB | Classroom Behavior Split | **94.58%** | **94.00%** | 

*Note: All baseline comparison values reported in the paper are directly quoted from their original published sources under standard protocols.*

## 🏗️ Model Architecture

```
               +-------------------------------------------------------+
               |                    SKELETON STREAM                    |
Skeleton ----> |  ST-GCN (10 Blocks, C=64->256, Dropout=0.3)           | --+
Sequences      |  Outputs: Global Pose Vector & Feature Map F_P        |   |
               +-------------------------------------------------------+   |
                                           |                               |
                                   Temporal Pooling                        |
                                           v                               v
               +-------------------------------------------------------+ +--------------------+
RGB Middle --> |                      RGB STREAM                       | |      CPIB          |
Frame Crop     |  Xception Backbone -> Spatial Context Map F_C         | | (Transformer       | --> Action
(299x299)      |                           |                           | |  Encoder, 3 L,   |     Logits
               |                           v                           | |  8 Heads, D=512)  |
               |  Context Feature Enhancement Block (CFEB)              | +--------------------+
               |  (Pose-guided Spatial Cross-Attention -> F_Ce)        |           ^
               +-------------------------------------------------------+           |
                                           |                                       |
                                           +---------------------------------------+

```

## 🛠️ Environment Setup & Installation

### Dependencies

* Python 3.9+ (3.10 / 3.11 recommended)

* PyTorch >= 1.12.0

* torchvision >= 0.13.0

* `timm` >= 0.6.12 (for pre-trained Xception backbone)

* `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `thop`, `pytorch-grad-cam`

### Installation Commands

```
# 1. Clone repository
git clone https://github.com/hieu79115/MMFF.git
cd MMFF

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install required packages
pip install --upgrade pip
pip install -r requirements.txt

```

## ⚙️ Hyperparameters & 3-Stage Training Recipe

All experiments follow a strict **three-stage sequential training schedule**:

| Hyperparameter / Setting | Stage 1: Skeleton Stream | Stage 2: RGB + CFEB | Stage 3: CPIB Fusion | 
| ----- | ----- | ----- | ----- | 
| **Active Target** | Skeleton Branch (`skel_encoder`) | RGB Branch (`rgb_encoder` + CFEB) | Joint Model (`fusion_head`) | 
| **Frozen Modules** | None | Skeleton Branch (`torch.no_grad()`) | None (Unfrozen) | 
| **Backbone Strategy** | — | Xception frozen initially for 10 epochs | Fully unfrozen | 
| **Epochs** | 100 | 100 | 100 | 
| **Batch Size** | 8 | 8 | 32 | 
| **Initial Learning Rate (**$LR_0$**)** | $1 \times 10^{-3}$ | $5 \times 10^{-4}$ | $1 \times 10^{-4}$ | 
| **Optimizer** | AdamW | AdamW | AdamW | 
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | 
| **LR Scheduler** | Cosine Annealing ($\eta_{min}=10^{-6}$) | Cosine Annealing ($\eta_{min}=10^{-6}$) | Cosine Annealing ($\eta_{min}=10^{-6}$) | 
| **Warmup** | 5 epochs (Linear) | 5 epochs (Linear) | 5 epochs (Linear) | 
| **Dropout** | 0.3 | 0.3 | 0.3 | 
| **Loss Function** | Cross-Entropy ($\epsilon=0.1$) | Cross-Entropy ($\epsilon=0.1$) | Cross-Entropy ($\epsilon=0.1$) / Focal Loss\* | 

*\*Focal Loss (*$\alpha=0.25, \gamma=2.0$*) is additionally applied on HCMUE-SB to address class imbalance.*

## 📁 Dataset Directory Layout

Data loading via `utils/dataset.py` expects preprocessed samples inside `--data_dir` (default: `./data`):

```
data/
├── train_data.pkl    # List of training dicts: video_name, label, normalized_skeleton, rgb_crop, etc.
├── test_data.pkl     # List of testing dicts
└── images/           # Raw/cropped RGB images (fallback)

```

**Skeleton Frame Configurations:**

* **UTD-MHAD**: 20 joints, resampled to $T = 32$ frames.

* **NW-UCLA**: 21 joints, resampled to $T = 32$ frames.

* **HCMUE-SB**: 13 joints (upper body), resampled to $T = 15$ frames.

## 🚀 Usage & Reproduction Commands

### 1. Verification with Dummy Data

Run an end-to-end pipeline check without real dataset files:

```
python test_pipeline.py

```

### 2. Full 3-Stage Training

#### UTD-MHAD Example:

```
# Stage 1: Train Skeleton Stream
python train.py --dataset utd --stage skeleton --batch_size 8 --lr 0.001 --epochs 100 --num_frames 32 --seed 42

# Stage 2: Fine-tune RGB Stream + CFEB (loads best skeleton weights)
python train.py --dataset utd --stage rgb --cross_attention normal --batch_size 8 --lr 0.0005 --epochs 100 --num_frames 32 --seed 42

# Stage 3: Train CPIB Fusion (loads skeleton & RGB weights)
python train.py --dataset utd --stage fusion --fusion_type cmaf --cross_attention normal --batch_size 32 --lr 0.0001 --epochs 100 --num_frames 32 --seed 42

```

#### NW-UCLA Example:

```
python train.py --dataset nw-ucla --stage skeleton --batch_size 8 --lr 0.001 --epochs 100 --num_frames 32 --seed 42
python train.py --dataset nw-ucla --stage rgb --cross_attention normal --batch_size 8 --lr 0.0005 --epochs 100 --num_frames 32 --seed 42
python train.py --dataset nw-ucla --stage fusion --fusion_type cmaf --cross_attention normal --batch_size 32 --lr 0.0001 --epochs 100 --num_frames 32 --seed 42

```

### 3. Multi-Seed Experiments

Run independent seeds (0 to 4) to compute mean and standard deviation:

```
python run_multiseed.py --seeds 0 1 2 3 4 --split_seed 42 \
  --dataset utd --stage fusion --fusion_type cmaf --cross_attention normal

```

*Generates `multiseed_summary.json` containing accuracy statistics.*

### 4. Model Evaluation & Confusion Matrix

Evaluate a trained checkpoint on held-out test data and generate high-resolution confusion matrix figures:

```
python test.py --dataset utd --stage fusion --fusion_type cmaf --cross_attention normal --batch_size 32 --seed 42

```

### 5. Grad-CAM Visualization

Generate spatial attention heatmaps before and after CFEB pose guidance:

```
python visualize_gradcam.py --dataset utd --data_dir ./data --model_path best_fusion_utd_cmaf_attn-normal_seed42.pth

```

*Outputs `gradcam_real_utd.png` showing contextual attention refinement.*

### 6. Inference Latency, Memory & FLOPs Benchmarking

Measure model FLOPs and parameter counts:

```
python check_model_stats.py --dataset utd --stage fusion --compare_all

```

Benchmark GPU inference latency, throughput, and peak memory usage:

```
python benchmark_inference.py --dataset utd --stage fusion --seed 42 --batch_size 1 --max_samples 100 --output benchmark_utd.json

```

## 📂 Repository Structure

```
MMFF/
├── models/
│   ├── attention.py          # CFEB (CrossModalAttention & Reversed variant)
│   ├── backbone.py           # RGB Stream (Pre-trained Xception + CFEB)
│   ├── fusion.py             # CPIB (FusionTransformer & baseline fusions)
│   ├── st_gcn.py             # Skeleton Stream (10 ST-GCN Blocks)
│   └── mmff_net.py           # Integrated PCINet Framework
├── utils/
│   ├── dataset.py            # PyTorch Dataset, temporal resampling & RGB crop loader
│   ├── graph.py              # Skeleton graph definitions (UTD, NW-UCLA, HCMUE-SB, NTU)
│   ├── losses.py             # Focal Loss & Label-Smoothing Cross-Entropy implementation
│   └── model_stats.py        # Parameters & FLOPs counting utilities (thop integration)
├── config.py                 # Central hyperparameter configuration
├── train.py                  # 3-stage training script
├── test.py                   # Evaluation script & confusion matrix plotter
├── run_multiseed.py          # Automated multi-seed training & statistical evaluation
├── benchmark_inference.py    # Latency, throughput, and GPU memory benchmark
├── check_model_stats.py      # Command-line utility to inspect model parameters/FLOPs
├── visualize_gradcam.py      # Grad-CAM visualization for CFEB spatial attention
├── requirements.txt          # Python dependency versions
├── test_pipeline.py          # Quick dummy end-to-end pipeline verification
└── README.md                 # Documentation

```
