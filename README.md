# ASFM-Net-Review

***Status: Done***

This is an unofficial implement of ASFM-Net, and we hope it works orz.

## Preparation

***pytorch > 1.10.1 required!***

1. Complie Chamfer3D, code from https://github.com/AllenXiangX/SnowflakeNet
```
cd Chamfer3D
python setup.py --user install
```
2. Complie pointnet++ which includes fps_sampling we need, code from https://github.com/AllenXiangX/SnowflakeNet
```
cd pointnet2_ops_lib
python setup.py --user install
```
3. Dataset configs at config_pcn.py, modify it refer to https://github.com/hzxie/GRNet

Note that I am using gradient accumulation to simulate a big batchsize(32 in paper), You can set `accumulation_steps = 1` in train_*.py under core directoroies to cancel this.

## Run
1. Train pretrined PCN.
```
python main_pcn.py --baseline
```
2. Modify __C.CONST.PCNWEIGHTS in config_pcn.py to direct to pre-trained models obtained in step1, train ASFM-net.
```
python main_pcn.py --backbone
```

## Results
Haven't reproduce the results
If you have any ***suggestions or ideas*** about my work, feel free to ***refer a issue even a pr*** to keep me informed, thanks.
