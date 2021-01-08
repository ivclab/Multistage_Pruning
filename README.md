# Multistage_Pruning

This is an official Pytorch implementation of [Pruning Depthwise Separable Convolutions for MobileNet Compression](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_WCCI_2020/IJCNN/Papers/N-21727.pdf) 

Created by Cheng-Hao Tu, Jia-Hong Lee, Yi-Ming Chan and Chu-Song Chen

The code is released for academic research use only. For commercial use, please contact Prof. Chu-Song Chen(chusong@csie.ntu.edu.tw).

## Introduction 
Deep convolutional neural networks are good at accuracy while bad at efficiency. To improve the inference speed, two directions have been explored in the past, lightweight model designing and network weight pruning. Lightweight models have been proposed to improve the speed with good enough accuracy. It is, however, not trivial if we can further speed up these “compact” models by weight pruning. In this paper, we present a technique to gradually prune the depthwise separable convolution networks, such as MobileNet, for improving the speed of this kind of “dense” network. When pruning depthwise separable convolutions, we need to consider more structural constraints to ensure the speedup of inference. Instead of pruning the model with the desired ratio in one stage, the proposed multi-stage gradual pruning approach can stably prune the filters with a finer pruning ratio. Our method achieves satisfiable speedup with little accuracy drop for MobileNets. 


## Prerequisites 
* python==3.6
* torch==1.4.0
* torchvision==0.5.0
* tqdm==4.31.1
* networkx==2.4
* graphviz==0.13
* [thop](https://github.com/Lyken17/pytorch-OpCounter)


## Usage 

### Preparing the data 

You will need to download the ImageNet dataset from its [official website](http://image-net.org/download) and place the downloaded images under `DATA/`. Please see the line 26, 27 in `utils/datasets.py` for details about how to set the path.


### Training the baseline models

Please use the following commands for training baseline MobileNetV1 and MobileNetV2, respectively. 

``` 
python train.py --config_name MobileNetV1_ImageNet --target_mode baseline
```

and 

``` 
python train.py --config_name MobileNetV2_ImageNet --target_mode baseline
```

The first command will train MobileNetV1 fraom scratch on ImageNet, and it may take a while. The trained baseline MobileNetV1 can be downloaded [here](https://drive.google.com/file/d/1SYzK8Hocwc_-uxiAYBeOzGg8eona8C8R/view?usp=sharing). 
Since Pytorch already provides pretrained MobileNetV2 on ImageNet, the second command simply loads the pretrained model and evaluate its accuracy.


### Multistage gradual pruning

Please use the following command for multistage gradual pruining the baseline models. Substitute {NETWORK} with MobileNetV1 or MobileNetV2 to prune on different networks, and substitute {PRUNE_STAGE} with 8stage_prune or 16stage_prune for various number of pruning stages. 

``` 
python train.py --config_name {NETWORK}_ImageNet --target_mode {PRUNE_STAGE}-magnitude
```

We provide the pruned MobileNetV1 with various pruning ratios as follows: 

| Model | Top-1 Accuracy | FLOPs | Params | 
|:---:|:---:|:---:|:---:|
| [MobileNetV1_1.00x](https://drive.google.com/file/d/1qXOdg-nTGKWR3-qnaFknN_bWUfoU0jWH/view?usp=sharing) | 70.69 | 579.8479 M | 4.2320 M |
| [MobileNetV1_0.75x](https://drive.google.com/file/d/1Cwl848hz_8IbRXAP70vnSFnLvgi3ux_x/view?usp=sharing) | 68.84 | 333.7325 M | 2.5856 M |
| [MobileNetV1_0.50x](https://drive.google.com/file/d/1Jbp_rPfWxo7OOi2NDVgYTfWi3zcS5Oim/view?usp=sharing) | 64.15 | 155.0518 M | 1.3316 M |
| [MobileNetV1_0.25x](https://drive.google.com/file/d/1TBKaYSwphzIt3StXBCC1ZAbebrR_dRN2/view?usp=sharing) | 51.62 |  43.8076 M | 0.4701 M |


### Evaluate the pruned models 

Please use the following command for evaluating the accuracy, number of flops and parameters for a pruned model. Substitute {NETWORK} with MobileNetV1 or MobileNetV2 and {PRUNE_STAGE} with 8stage_prune or 16stage_prune. The {PRUNE_RATIO} should be the pruning ratios should be the pruning ratios reached after each stage, for example, 0.125, 0.250, 0.375, ..., 0.875 for 8stage_prune. 

```
python evaluate_pruned_networks.py --network_name {NETWORK} --dataset_name ImageNet --target_mode {PRUNE_STAGE}-magnitude --prune_ratio {PRUNE_RATIO}
```

### Evaluate the unpruned models 

To evaluate the unpruned models (baseline models or models with 0.0 pruning ratio), please use the following command. 

```
python evaluate_unpruned_networks.py --network_name {NETWORK} --dataset_name ImageNet --chkpt_path {CHKPT_PATH}
```

The {NETWORK} could be MobileNetV1 or MobileNetV2, and {CHKPT_PATH} is the path of the target checkpoint. See the following example that evaluates the ImageNet-trained baseline MobileNetV1 provided above. Note that we place the downloaded checkpoint `baseline.pth` under `CHECKPOINTS/MobileNetV1_ImageNet/baseline/`. 

```
python evaluate_unpruned_networks.py --network_name MobileNetV1 --dataset_name ImageNet --chkpt_path CHECKPOINTS/MobileNetV1_ImageNet/baseline/baseline.pth
```

## Citation
Please cite following paper if these codes help your research:

    @inproceedings{tu2020pruning,
    title={Pruning Depthwise Separable Convolutions for MobileNet Compression},
    author={Tu, Cheng-Hao and Lee, Jia-Hong and Chan, Yi-Ming and Chen, Chu-Song},
    booktitle={2020 International Joint Conference on Neural Networks (IJCNN)},
    pages={1--8},
    year={2020},
    organization={IEEE}
    }

## Contact 
Please feel free to leave suggestions or comments to Cheng-Hao Tu(andytu28@iis.sinica.edu.tw), Jia-Hong Lee(honghenry.lee@iis.sinica.edu.tw), Yi-Ming Chan(yiming@iis.sinica.edu.tw), Chu-Song Chen(chusong@csie.ntu.edu.tw)
