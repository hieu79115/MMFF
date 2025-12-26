**Overview**
- MMFF (Multi-Modal Fusion Framework) is an action recognition model that fuses Skeleton (ST-GCN) and RGB (Xception) streams, then applies a Transformer-based late fusion to learn interactions between modalities.
- Repo supports both real data (exported `.npy/.pkl`) and dummy data (quick pipeline checks).

**Architecture**
- Skeleton stream: ST-GCN extracts spatio-temporal features and returns
	- a 256-d vector, and
	- a feature map used to guide the RGB stream via Cross-Modal Attention.
- RGB stream: `Xception` backbone (from `timm`, pretrained on ImageNet) + Cross-Modal Attention; global average pooled to a 2048-d vector.
- Late fusion: `TransformerEncoder` with a `[CLS]` token mixes the two vectors and performs classification.
- Key sources: [models/mmff_net.py](models/mmff_net.py), [models/st_gcn.py](models/st_gcn.py), [models/backbone.py](models/backbone.py), [models/attention.py](models/attention.py), [models/fusion.py](models/fusion.py).

**Requirements**
- Python 3.10+ (3.10 or 3.11 recommended).
- PyTorch and TorchVision (versions pinned in [requirements.txt](requirements.txt)).
- `timm` to load `legacy_xception` (pretrained=True will fetch ImageNet weights on first run).
- Windows supported (commands below use PowerShell/CMD). Linux/Mac are similar.

**Setup**
1) Create a virtual environment and install dependencies:
```
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) Optional: verify PyTorch CUDA availability:
```
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Data Structure**

The dataset loader [utils/dataset.py](utils/dataset.py) expects exported files under `./data`:
- `train_data.npy` + `train_label.pkl`: training pool
- `test_data.npy` + `test_label.pkl`: held-out test set
- `images/`: RGB frames (one image per sample name from the `.pkl`)

During training, validation is a deterministic split from the training pool:
- `mode='train'` and `mode='val'` both read from `train_*` and split using `--val_ratio` + `--split_seed`.
- `mode='test'` reads from `test_*` (and falls back to legacy `val_*` if present).

Important defaults in [utils/dataset.py](utils/dataset.py):
- Fixed frames: 32 (`num_frames=32`).
- RGB image size: 299×299 (Xception-friendly).
- Joints: NTU=25, UTD=20.

## How to Run

### 1. Quick Sanity Check (End-to-End Pipeline)
Run the complete pipeline with dummy data to verify setup:
```
python test_pipeline.py
```
**Expected output:** prints skeleton shape, RGB shape, and model output shape.

### 2. Training
Training is stage-wise (each stage saves its own best checkpoint):
- `skeleton`: train skeleton stream
- `rgb`: train RGB stream (optionally warm-start from skeleton)
- `fusion`: train final fusion head (warm-start from skeleton + rgb if available)

Examples:
```
# 1) Skeleton stage
python train.py --dataset ntu --stage skeleton --epochs 30 --batch_size 8

# 2) RGB stage
python train.py --dataset ntu --stage rgb --epochs 30 --batch_size 8

# 3) Fusion stage
python train.py --dataset ntu --stage fusion --epochs 30 --batch_size 8
```

Key training options:
- `--dataset`: dataset name (default: `ntu`)
- `--stage`: `skeleton` | `rgb` | `fusion` (default: `fusion`)
- `--epochs`, `--batch_size`, `--lr`
- `--val_ratio`: validation ratio split from train pool (default: `0.1`)
- `--split_seed`: seed for deterministic train/val split (default: `42`)

Outputs:
- Best weights: `best_{stage}_{dataset}.pth`
- Training history plot: `history_{stage}_{dataset}.png`

### 3. Evaluation

Evaluate a trained checkpoint:
```
python test.py --dataset ntu --stage fusion --batch_size 4
python test.py --dataset utd --stage fusion --batch_size 4
```

#### Key Evaluation Options:
- `--dataset`: `ntu` or `utd` (default: `ntu`)
- `--stage`: which checkpoint to evaluate (`skeleton` | `rgb` | `fusion`)
- `--batch_size`: batch size (default: 4)
- `--is_dummy`: use dummy data (accuracy will be random)

**Note:** With `is_dummy=True`, accuracy will be random (for pipeline testing only). When real data is integrated, the script will also generate `confusion_matrix_{dataset}.png`.

### 4. Full Training Workflow Example
```bash
# 1. Quick sanity check
python test_pipeline.py

# 2. Train the model
python train.py --dataset ntu --stage skeleton
python train.py --dataset ntu --stage rgb
python train.py --dataset ntu --stage fusion

# 3. Evaluate the model on held-out test set
python test.py --dataset ntu --stage fusion
```

**Additional Information**

### Python and CUDA Check
Before running, verify your environment:
```
python --version
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Expected Inputs (for real data)
- Skeleton: tensor `(N, 3, T=32, V)` where `V=25 (NTU)` or `V=20 (UTD)`.
- RGB: tensor `(N, 3, 299, 299)` (apply ImageNet normalization if using pretrained).
- `MMFFDataset` returns a 4-tuple: `(skeleton_feat, rgb_img, 0, label)` where the 3rd element is a placeholder.

**Outputs**
- Best weights: `best_{stage}_{dataset}.pth`.
- Training plots: `history_{stage}_{dataset}.png`.
- Confusion matrix (when not using dummy): `confusion_matrix_{dataset}.png`.

**Troubleshooting**
- Failing to download `Xception` weights from `timm`:
	- Ensure Internet on first run; or set `pretrained=False` in [models/backbone.py](models/backbone.py).
- OOM or memory pressure: reduce `--batch_size` and/or use CPU.
- `test.py` cannot find weights: run training first to produce `best_{stage}_{dataset}.pth`.
- If your held-out files are named `val_*`, rename to `test_*` or keep them; loader falls back to legacy `val_*` automatically.

**Next Steps (TODO)**
- Add preprocessing (skeleton normalization, RGB resize/crop) in [utils/preprocess.py](utils/preprocess.py) as needed.
- Potential improvements: deeper Transformer for fusion, RGB augmentations, ST-GCN regularization.

**References & Inspiration**
- ST-GCN for skeleton-based action recognition.
- Xception (ImageNet) as the RGB backbone via `timm`.
- Transformer-based late fusion with a `[CLS]` token.

**Contact & Feedback**
- For issues/bugs: open an issue with a minimal repro and your Python/PyTorch versions.