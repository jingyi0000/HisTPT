# Historical Test-time Prompt Tuning for Vision Foundation Models (NeurIPS 2024)

### [Paper](https://arxiv.org/pdf/2410.20346)

![overview](docs/histpt.png)

## Installation

```sh
# Python Package Installation
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_custom.txt
```

## Data Preparation
Please refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/datasets) for data preparation.

### Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/):

```
└── dataset
    ├── cityscapes
        ├── annotations
        ├── gtFine
        └── leftImg8bit
```


## Test-Time Tuning

## Citation
We appreciate your citations if you find our paper related and useful to your research!
```
@article{zhang2024historical,
  title={Historical test-time prompt tuning for vision foundation models},
  author={Zhang, Jingyi and Huang, Jiaxing and Zhang, Xiaoqin and Shao, Ling and Lu, Shijian},
  journal={arXiv preprint arXiv:2410.20346},
  year={2024}
}
```


## Acknowledgments
This code is heavily borrowed from [[SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)].
