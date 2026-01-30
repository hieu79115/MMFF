# NW-UCLA Dataset Support Implementation Summary

## Overview
Successfully added support for the NW-UCLA (Northwestern-UCLA Multiview Action 3D) dataset with 21 skeleton joints and 10 action classes to the MMFF framework.

## Changes Made

### 1. Configuration (config.py)
- Added NW-UCLA to `NUM_CLASSES` dictionary: `'nw-ucla': 10`
- Uses existing training configurations (epochs, learning rates) via `Config.get_num_classes()` method

### 2. Skeleton Graph Structure (utils/graph.py)
- Added 21-joint skeleton configuration for NW-UCLA
- Defined skeleton topology with:
  - 21 joints (Kinect v1 + computed neck joint)
  - 41 edges (21 self-links + 20 neighbor links)
  - Proper adjacency matrix (3, 21, 21)
- Joint structure includes:
  - Spine: hip center, spine, shoulder center, head
  - Arms: left/right shoulder, elbow, wrist, hand
  - Legs: left/right hip, knee, ankle, foot
  - Computed joint: neck

### 3. Dataset Loader (utils/dataset.py)
- Added logic to set `num_joints = 21` when `dataset_name == 'nw-ucla'`
- Automatic skeleton normalization handles 21 joints correctly
- Works with both dummy and real data

### 4. Training Script (train.py)
- Updated to use `Config.get_num_classes(args.dataset)` instead of hardcoded values
- Automatically supports NW-UCLA through config

### 5. Evaluation Script (test.py)
- Added 'nw-ucla' to dataset choices
- Added 10 class names for NW-UCLA:
  1. pick_up_with_one_hand
  2. pick_up_with_two_hands
  3. drop_trash
  4. walk_around
  5. sit_down
  6. stand_up
  7. donning
  8. doffing
  9. throw
  10. carry
- Updated to use `Config.get_num_classes()` for dynamic class count

### 6. Documentation (README.md)
- Added NW-UCLA to supported datasets list
- Updated joint count documentation: "NTU=25, UTD=20, NW-UCLA=21"
- Added training examples for NW-UCLA
- Added evaluation examples for NW-UCLA
- Added test script documentation

### 7. Test Script (test_nw_ucla.py)
- Created comprehensive test script to verify all components
- Tests include:
  - Configuration validation
  - Graph structure validation
  - Dataset functionality
  - Model architecture
  - Training setup
  - Class names
- All tests pass successfully

## Model Compatibility
The existing model architecture (st_gcn.py, mmff_net.py) automatically supports NW-UCLA through the `dataset` parameter, which is passed to the Graph class. No changes were needed to the model code.

## Usage

### Training
```bash
python train.py --dataset nw-ucla --stage skeleton --batch_size 16
python train.py --dataset nw-ucla --stage rgb --batch_size 16
python train.py --dataset nw-ucla --stage fusion --batch_size 16
```

### Evaluation
```bash
python test.py --dataset nw-ucla --stage fusion --batch_size 4
```

### Testing Configuration
```bash
python test_nw_ucla.py
```

## Files Modified
1. config.py - Added NW-UCLA class count
2. utils/graph.py - Added 21-joint skeleton structure
3. utils/dataset.py - Added joint count logic
4. train.py - Updated for dynamic class count
5. test.py - Added class names and dataset choice
6. README.md - Added documentation
7. test_nw_ucla.py - Created (new file)

## Verification
All components have been tested with dummy data and work correctly:
- ✅ Configuration loads 10 classes for NW-UCLA
- ✅ Graph creates 21-joint skeleton with correct topology
- ✅ Dataset produces (3, 32, 21) skeleton tensors
- ✅ Model accepts 21-joint input and produces correct output shape
- ✅ Training loop works with NW-UCLA configuration
- ✅ Class names defined for confusion matrix

## Technical Details

### Joint Count Handling
The framework automatically handles different joint counts per dataset:
- NTU RGB+D: 25 joints
- UTD-MHAD: 20 joints  
- NW-UCLA: 21 joints

### Skeleton Graph Topology
The NW-UCLA skeleton follows Kinect v1 topology with an additional computed neck joint:
- Central spine: hip center → spine → shoulder center → head
- Neck joint connects to shoulder center
- Symmetric arms and legs branching from center

### Data Format
Expected skeleton data format: (C, T, V) where:
- C = 3 (x, y, z coordinates)
- T = 32 (resampled frames)
- V = 21 (joints for NW-UCLA)

## Next Steps
To use with real NW-UCLA data:
1. Prepare data in the expected format (train_data.pkl, test_data.pkl)
2. Ensure skeleton data has 21 joints in the correct order
3. Run training with appropriate batch size and epochs
4. Evaluate with test.py to get accuracy and confusion matrix
