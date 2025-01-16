# Historical Test-time Prompt Tuning for Vision Foundation Models (NeurIPS 2024)

### [Paper](https://arxiv.org/pdf/2410.20346)

![overview](docs/arch.pdf)

## Installation and Data Preparation

Please refer to [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) for installation and [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/datasets) for data preparation.


## Test-time Tuning

```sh
bash scripts/train.sh
```

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
This code is heavily borrowed from [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once).
