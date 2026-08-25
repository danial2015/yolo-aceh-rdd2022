# BL-YOLOv8-S replication

This implementation reproduces the proposed BL-YOLOv8 architecture from Wang et al., *BL-YOLOv8: An Improved Road
Defect Detection Model Based on YOLOv8* (Sensors, 2023), using the current Ultralytics codebase.

The model definition is [bl-yolov8s.yaml](../../../ultralytics/cfg/models/v8/bl-yolov8s.yaml). It remains a
YOLOv8-S model; it is not a YOLO12 port.

## Proposed-method components

| Paper component | Implementation | YAML layers |
| --- | --- | --- |
| SimSPPF | Three serial 5x5 max-pooling operations with Conv-BN-ReLU before and after concatenation | 9 |
| LSK-Attention | Depthwise 5x5 plus dilated depthwise 7x7 selective spatial attention | 10 |
| Weighted BiFPN | Non-negative normalized learnable weights: ReLU(w_i) / (sum ReLU(w_i) + 1e-4) | 15, 18, 21, 24, 27 |

The BiFPN graph includes the paper's P2-to-P3 injection (layer 20), produces P3/P4/P5 detection features, and its
five fusion nodes have respectively 2, 2, 3, 3, and 2 learnable scalar weights. At nc=5, the model builds with
7,808,465 parameters; the small difference from the paper's rounded 7.82 M value depends on detection class count
and the current Ultralytics implementation.

## Training comparison

Use the supplied Kaggle notebook
[kaggle_bl_yolov8s_rdd2022.ipynb](../../../examples/kaggle_bl_yolov8s_rdd2022.ipynb). It:

1. clones this branch and installs the repository source;
2. checks and prints the model graph before training;
3. transfers only structurally identical YOLOv8-S pretrained backbone tensors (layers 0-8) from yolov8s.pt;
4. uses the paper's Table 3 optimizer settings: SGD, learning rate 0.01, momentum 0.937, weight decay 0.0005,
   160 epochs, 640 image size, and batch 64;
5. validates best.pt on validation and test sets, then writes a results ZIP.

For a fair baseline, train the official yolov8s.yaml on exactly the same split, seed, image size, augmentation,
pretrained initialization policy, and hyperparameters. The paper reports filtering its China dataset to 4,373 images;
the notebook prints the actual count and warns if the supplied dataset differs. It keeps the requested five classes
(D00, D10, D20, D40, Repair), so reported results should be described as a replication experiment on this dataset
configuration, not a direct replacement for the paper's reported result.
