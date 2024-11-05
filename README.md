# Feature-metric registration with Images

All credits to the beautiful people behind this repository: https://github.com/XiaoshuiHuang/fmr

Run the code using the steps below.
If you only want to use the FMR algorithm without building your own training and evaluation data, run step 1 first. Then skip to step 4 and continue from there.

PLY FORMAT ENCOURAGED

### 1. Install dependencies:

```
pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html argparse numpy glob matplotlib six
```

### 2. Train the model

2.1. Train on dataset ModelNet40:

```
python train.py -data modelnet
```

2.2. Train on dataset 7scene:

```
python train.py -data 7scene
```

### 3. Evalute the model

3.1. Evaluate on dataset ModelNet40:

```
python evalute.py -data modelnet
```

3.2. Evaluate on dataset 7scene:

```
python evalute.py -data 7scene
```

The pretrained models are stored in the result folder.

### 4. Code for testing your own point clouds

4.1 Run below for an example of what the algorithm does:

```
python fmr-demo.py
```

4.2. Run below for help in using additional features of the program:

```
python fmr-demo.py -h
```

### Citation and Acknowledgements

```
@InProceedings{Huang_2020_CVPR,
    author = {Huang, Xiaoshui and Mei, Guofeng and Zhang, Jian},
    title = {Feature-Metric Registration: A Fast Semi-Supervised Approach for Robust Point Cloud Registration Without Correspondences},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

We would like to thank the open-source code of [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet) and [pointnetlk](https://github.com/hmgoforth/PointNetLK)
