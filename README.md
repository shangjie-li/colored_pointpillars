# colored_pointpillars

Implementation of Colored PointPillars in PyTorch for KITTI 3D Object Detetcion

## Acknowledgement
 - This repository references [open-mmlab](https://github.com/open-mmlab/OpenPCDet)'s work.

## Installation
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/colored_pointpillars.git
   ```
 - Install PyTorch environment with Anaconda (Tested on Ubuntu 16.04 & CUDA 10.2)
   ```
   conda create -n pcdet.v0.5.0 python=3.6
   conda activate pcdet.v0.5.0
   cd colored_pointpillars
   pip install -r requirements.txt
   ```
 - Install spconv
   ```
   # Try the command below:
   pip install spconv-cu102
   
   # If there is `ERROR: Cannot uninstall 'certifi'.`, try:
   pip install spconv-cu102 --ignore-installed
   ```
 - Compile external modules
   ```
   cd colored_pointpillars
   python setup.py develop
   ```
 - Install visualization tools
   ```
   pip install mayavi
   pip install pyqt5
   
   # If you want import opencv, run:
   pip install opencv-python
   
   # If you want import open3d, run:
   pip install open3d-python
   ```

## KITTI3D Dataset (41.5GB)
 - Download KITTI3D dataset: [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip), [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip), [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) and [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip).
 - Download [road plane](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) for data augmentation.
 - Organize the downloaded files as follows
   ```
   colored_pointpillars
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2 & planes
   │   │   │── testing
   │   │   │   ├──calib & velodyne & image_2
   ├── layers
   ├── utils
   ```
 - Generate the ground truth database and data infos by running the following command
   ```
   # This will create gt_database dir and info files in colored_pointpillars/data/kitti.
   cd colored_pointpillars
   python -m data.kitti_dataset create_kitti_infos data/config.yaml
   ```
 - Display the dataset
   ```
   # Display the training dataset with data augmentation
   python dataset_player.py --training --data_augmentation
   
   # Display the testing dataset
   python dataset_player.py
   ```

## Demo
 - Run the demo with a pretrained model
   ```
   # Run on a single file
   python demo.py --ckpt=path_to_your_ckpt --data_path=data/kitti/training/velodyne/000008.bin
   
   # Run on a folder
   python demo.py --ckpt=path_to_your_ckpt --data_path=data/kitti/training/velodyne
   ```

## Training
 - Run the command below to train
   ```
   python train.py --batch_size=2
   ```

## Evaluation
 - Run the command below to evaluate
   ```
   python test.py --ckpt=path_to_your_ckpt
   ```
