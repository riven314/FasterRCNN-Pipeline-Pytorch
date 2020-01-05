### FasterRCNN-Pipeline-Pytorch
Setup a pipeline for training (transfer learning) Faster-RCNN in PyTorch. Data are in VOC format

### Setup
1. install Cython: ```pip install Cython```
2. install pycocotools: ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```
3. install easydict: ```pip install easydict```
4. install pytorch and torchvision: ```conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch```
5. install tensorflow (for tensorboardX): ```conda install -c conda-forge tensorflow```)
6. install tensorboardX: ```pip install tensorboardX``` 

### Remarks
- there is a minor adjustment on vision/references/detection/engine.py (on ```train_one_epoch```)
- PIL.Image causes a lot of trouble when working with opencv, torchvision. Suggest to use cv2 (at least on Windows)
- test model on easier samples first for first gate testing