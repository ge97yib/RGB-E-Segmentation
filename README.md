# RGBX Semantic Segmentation

This repository contains the implementation of **multi-modal semantic segmentation** using RGB and event camera data. The project is based on my Master's thesis work:  
> *Multi-Modal Fusion of Image Sequences for Dense Prediction with RGB and Event Cameras in Autonomous Driving*

## Project Overview

This research explores the integration of RGB and event camera data to enhance dense prediction tasks—especially semantic segmentation—in autonomous driving. RGB cameras deliver high-resolution visual information essential for scene understanding, while event cameras provide high temporal resolution and dynamic range, enabling motion detection and perception under extreme lighting conditions.  

By fusing these complementary modalities, the system constructs a more complete and resilient representation of dynamic traffic environments. This work investigates multi-modal feature fusion strategies tailored for semantic segmentation.

## Project Structure

```
RGB_E_Segmentation/
├── config*.py              # Various configuration files for different datasets
├── eval.py                # Evaluation entry point
├── requirements.txt       # Required Python packages
├── _train*.py             # Training scripts with various experiment settings
├── dataloader/            # Dataset wrappers and preprocessing
├── engine/                # Training engine, evaluator, logger
├── models/                # Encoders, decoders, and fusion modules
├── utils/                 # Losses, metrics, LR scheduler, visualization tools
```

## Getting Started

### Installation

```bash
# Clone this repository
git clone https://github.com/ge97yib/RGB-E-Segmentation.git
cd RGB_E_Segmentation

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation
The dataset folder would be structured like:
```bash
    <datasets>
    |-- <DatasetName1>
        |-- <RGBFolder>
            |-- <name1>.<ImageFormat>
            |-- <name2>.<ImageFormat>
            ...
        |-- <ModalXFolder>
            |-- <name1>.<ModalXFormat>
            |-- <name2>.<ModalXFormat>
            ...
        |-- <LabelFolder>
            |-- <name1>.<LabelFormat>
            |-- <name2>.<LabelFormat>
            ...
        |-- train.txt
        |-- test.txt
```

In train.txt and test.txt, please save the items as
```bash
    <name1>
    <name2>
```

Place your dataset in a structured directory and modify the corresponding config file (e.g. `config_nyudepth.py`, `config_dsec.py`, etc.) to point to your data path.

### Training

```bash
python _train.py --config config_nyudepth.py
```

### Evaluation

```bash
python eval.py -e "epoch"
```

<!-- ## Fusion Strategies

Implemented fusion modules include:

- Attention-based RGB-event fusion
- AdaIN-style feature alignment
- Dual-stream encoder-decoder architectures -->

## Supported Datasets

- DDD17
- DSEC
- MFNet
- DELIVER
- NYUDv2
- SUN RGB-D

Modify the config files accordingly to adapt to your target dataset.

## Acknowledgements

This project builds upon ideas and code from the following repositories:

- [Cascaded RGB-D Salient Object Detection (JingZhang617)](https://github.com/JingZhang617/cascaded_rgbd_sod)
- [RGBX Semantic Segmentation (huaaaliu)](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)

We thank the original authors for their contributions and open-sourcing their work.



<!-- ## Citation

If you use this work in your research or derive from it, please consider citing:

```
Mengyu Li. "Multi-Modal Fusion of Image Sequences for Dense Prediction with RGB and Event Cameras in Autonomous Driving." Master's Thesis, Technical University of Munich, 2025.
```  -->

<!-- ## License

This repository contains modified components from other open-source projects. Please check individual files for license information where applicable. All modifications made by Mengyu Li are released under [MIT License](https://opensource.org/licenses/MIT). -->

---

## Contact

For questions, please feel free to reach out via [GitHub Issues](https://github.com/ge97yib/RGB-E-Segmentation/issues) or email.
