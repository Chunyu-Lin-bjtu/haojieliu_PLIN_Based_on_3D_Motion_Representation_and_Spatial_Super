# paper
Pseudo LiDAR Point Cloud Interpolation Based on 3D Motion Representation and Spatial Supervision
## Notes
Our network is trained with the KITTI dataset alone, without pretraining on Cityscapes or other similar driving dataset (either synthetic or real). 
## Requirements
This code was tested with Python 3 and PyTorch 1.0 on Ubuntu 16.04,chamferdist.
- Install [PyTorch](https://pytorch.org/get-started/locally/) on a machine with CUDA GPU.
- The code for self-supervised training requires [OpenCV](http://pytorch.org/) along with the contrib modules. For instance,


- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset and the corresponding RGB images. 
- The code, data and result directory structure is shown as follows
```

├── data
|   ├── kitti_depth
|   |   ├── train
|   |   ├── val_selection_cropped
|   └── kitti_rgb
|   |   ├── train
|   |   ├── val_selection_cropped
├── results
```

## Training and testing
A complete list of training options is available with 
```bash
python main.py -h
```
For instance,
```bash
python main.py --train-mode dense -b 1 # train with the KITTI semi-dense annotations and batch size 1
python main.py --resume [checkpoint-path] # resume previous training
python main.py --evaluate [checkpoint-path] # test the trained model
eg.
CUDA_VISIBLE_DEVICES=1 python main.py --train-mode dense -b 1
python main.py --evaluate ./model_best.pth.tar 
```

## Questions
Please create a new issue for code-related questions. Pull requests are welcome.

## Citation
If you use our code or method in your work, please cite the following:
```bash
@article{liu2021pseudo,
	title={Pseudo-LiDAR Point Cloud Interpolation Based on 3D Motion Representation and Spatial Supervision},
	author={Liu, Haojie and Liao, Kang and Lin, Chunyu and Zhao, Yao and Guo, Yulan},
	journal={IEEE Transactions on Intelligent Transportation Systems},
	year={2021},
	publisher={IEEE}
	}
```

