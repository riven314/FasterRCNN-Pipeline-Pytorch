### FasterRCNN-Pipeline-Pytorch
Setup a pipeline for training (transfer learning) Faster-RCNN in PyTorch. Data are in VOC format

### Setup
1. install Cython: ```pip install Cython```
2. install pycocotools: ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```

*pycocotools is necessary for modules in vision

### Remarks
- there is a minor adjustment on vision/references/detection/engine.py (on ```train_one_epoch```)