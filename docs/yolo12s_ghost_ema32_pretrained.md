# Verified pretrained training: YOLO12s + GhostConv + EMA-32

The official `yolo12s.pt` checkpoint is stored by Ultralytics as an inference checkpoint. During checkpoint stripping, its parameters are marked `requires_grad=False`; an inference `model.info()` can therefore report `0 gradients`. This is a parameter-status count, not the numerical gradient from backpropagation and not a model-quality metric.

Use [the verified Kaggle notebook](../examples/kaggle_yolo12s_ghost_ema32_pretrained_verified_rdd2022.ipynb) for the GhostConv + EMA-32 experiment. It provides these guarantees before training begins:

- compatible official YOLO12s tensors are copied after accounting for the EMA layer's index shift;
- the two changed downsampling layers receive pretrained filters in their GhostConv primary branches and identity-initialized cheap branches;
- all target parameters other than YOLO's fixed DFL layer are marked trainable;
- the custom trainer receives the exact transferred model object instead of rebuilding a random model from YAML;
- gradients for a pretrained backbone layer, EMA, and both GhostConv branches are checked immediately before the first optimizer step.

The class-logit convolution of the COCO `80`-class source checkpoint cannot be copied to the target's five RDD2022 classes. Those three class-output layers remain randomly initialized by design. The shared backbone, box-regression branch, and compatible classification-branch tensors are still transferred.

The notebook stores the transfer report and first-step gradient norms inside the result ZIP. A failure in either verification stops training rather than producing a misleading mAP result.
