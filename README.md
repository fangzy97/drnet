# DRNet

This code is for the paper "DRNet: Double Recalibration Network for Few-Shot Semantic Segmentation". The implementation of the DRNet* is in [drnetv2](https://github.com/fangzy97/drnetv2)

## Usage

### Dependencies

* Python 2.7
* Pytorch 1.3.1

The dependencies can be installed by running:
```
pip install -r requirements.txt
```

### Dataset

* Pascal-5<sup>i</sup>: [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
* COCO-20<sup>i</sup>: [COCO2014](https://cocodataset.org/#download)

### Models

We provide trained Models with the ResNet-50 backbone on Pascal-5<sup>i</sup> and COCO-20<sup>i</sup> for performance evalution. You can download from [here](https://drive.google.com/file/d/1uB0nYj6iEUmQAstBsLusXaFSRTF2-h_w/view?usp=sharing).

### Scripts

* Change the dataset path in [db_path.py](db_path.py), [ss_datalayer.py](ss_datalayer.py) line 342 and [train_frame_coco.py](train_frame_coco.py).
* Pascal-5<sup>i</sup>
  * Run the train scripts
    ```
    cd scripts
    ./train_groupX.sh
    ```
  * Run the evalution scripts for Pascal-5<sup>i</sup>
    ```
    python test_frame_all.py
    ``` 
* COCO-20<sup>i</sup>
  * Run the train scripts
    ```
    cd scripts
    ./coco_train_groupX.sh
    ```
  * Run the evalution scripts for Pascal-5<sup>i</sup>
    ```
    python test_frame_all_coco.py
    ``` 