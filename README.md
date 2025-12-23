**Overview**
- MMFF (Multi-Modal Fusion Framework) is an action recognition model that fuses Skeleton (ST-GCN) and RGB (Xception) streams, then applies a Transformer-based late fusion to learn interactions between modalities.
- Current repo status: ready to run with dummy data for a full pipeline check. Real data loading is intentionally left as TODO in [utils/dataset.py](utils/dataset.py).

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
python - <<"PY"
import torch; print("CUDA:", torch.cuda.is_available())
PY
```

**Data Structure (Guidance)**
- Data folders (not used yet because real loader is pending):
	- [data/raw_skeleton](data/raw_skeleton)
	- [data/raw_video](data/raw_video)
- Important defaults in [utils/dataset.py](utils/dataset.py):
	- Fixed frames: 32 (`num_frames=32`).
	- RGB image size: 299×299 (Xception-friendly).
	- Joints: NTU=25, UTD=20.
- Current status: only `is_dummy=True` is supported, which generates random data for pipeline testing and quick benchmarking.

## How to Run

### 1. Quick Sanity Check (End-to-End Pipeline)
Run the complete pipeline with dummy data to verify setup:
```
python test_pipeline.py
```
**Expected output:** prints skeleton shape, RGB shape, and model output shape.

### 2. Training

#### Train on NTU Dataset (60 classes):
```
python train.py --dataset ntu --epochs 10 --batch_size 4 --is_dummy
```

#### Train on UTD Dataset (27 classes):
```
python train.py --dataset utd --epochs 10 --batch_size 4 --is_dummy
```

#### Key Training Options:
- `--dataset`: `ntu` or `utd` (default: `ntu`)
- `--epochs`: number of training epochs (default: 10)
- `--batch_size`: batch size (default: 4)
- `--lr`: learning rate (default: 1e-4)
- `--is_dummy`: use dummy data for testing (default: False, but required for now)

**Example with custom hyperparameters:**
```
python train.py --dataset ntu --epochs 20 --batch_size 8 --lr 5e-5 --is_dummy
```

**Outputs:**
- Best weights: `best_model_ntu.pth` or `best_model_utd.pth`
- Training history plot: `history_ntu.png` or `history_utd.png`

### 3. Evaluation

Evaluate a trained checkpoint:
```
python test.py --dataset ntu --batch_size 4 --is_dummy
python test.py --dataset utd --batch_size 4 --is_dummy
```

#### Key Evaluation Options:
- `--dataset`: `ntu` or `utd` (default: `ntu`)
- `--batch_size`: batch size (default: 4)
- `--is_dummy`: use dummy data (default: False, but required for now)

**Note:** With `is_dummy=True`, accuracy will be random (for pipeline testing only). When real data is integrated, the script will also generate `confusion_matrix_{dataset}.png`.

### 4. Full Training Workflow Example
```bash
# 1. Quick sanity check
python test_pipeline.py

# 2. Train the model
python train.py --dataset ntu --epochs 10 --is_dummy

# 3. Evaluate the model
python test.py --dataset ntu --is_dummy
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
- Best weights: `best_model_{dataset}.pth`.
- Training plots: `history_{dataset}.png`.
- Confusion matrix (when not using dummy): `confusion_matrix_{dataset}.png`.

**Troubleshooting**
- Failing to download `Xception` weights from `timm`:
	- Ensure Internet on first run; or set `pretrained=False` in [models/backbone.py](models/backbone.py).
- OOM or memory pressure: reduce `--batch_size` and/or use CPU.
- `test.py` cannot find weights: run training first to produce `best_model_{dataset}.pth`.
- `is_dummy=False` is not implemented yet (real data loader pending). Keep `--is_dummy` for now.

**Next Steps (TODO)**
- Implement real data loader in [utils/dataset.py](utils/dataset.py#L52-L78) (`_get_real_item`) and `__len__`.
- Add preprocessing (skeleton normalization, RGB resize/crop) in [utils/preprocess.py](utils/preprocess.py) as needed.
- Potential improvements: deeper Transformer for fusion, RGB augmentations, ST-GCN regularization.

**References & Inspiration**
- ST-GCN for skeleton-based action recognition.
- Xception (ImageNet) as the RGB backbone via `timm`.
- Transformer-based late fusion with a `[CLS]` token.

**Contact & Feedback**
- For issues/bugs: open an issue with a minimal repro and your Python/PyTorch versions.