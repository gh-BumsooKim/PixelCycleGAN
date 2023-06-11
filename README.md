# PixelCycleGAN
Unofficial Pytorch implementation of ["A Pixel image generation algorithm based on CycleGAN"](https://ieeexplore.ieee.org/document/9482118)

![image](assets/Overview.png)


## Experimental Environments

- os : ubuntu
- gpu : 8 X A100 (80G)
- pytorch : 1.12

## Implementation

***Because the dataset link in paper was broken or unavailable due to commercial usage, we manually collected our custom unpaired cartoon-pixel dataset in internet.***


| Dataset | # imgs | Query | Web site |
|:---:|:---:|:---:|:---:|
| (C) Cartoon | 2,000 | {*'cartoon'*, *'cartoon image'*} | Google, shutterstuck
| (P) Pixel Art | 2,000 | {*'pixel art'*, *'pixel art image'*} | Google



## Results

[Papers]

*TBA*

[Our Implementation]

*TBA*

## Train

```bash
python train.py
```


## Test

```bash
python test.py
```


## Acknowledge

Since this paper mainly based on CycleGAN, we also mainly reference the [Pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master) repository.