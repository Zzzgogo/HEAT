# HEAT for Active WSODD
Official Pytorch implementation for the paper titled "Hierarchical Evidence Aggregation in Two-dimensions for Active Water Surface Object Detection"

# Abstact
Water Surface Object Detection (WSOD) has achieved remarkable success within the realm of deep learning. However, one persistent challenge is the requirement for a substantial number of labeled samples. As a solution to this issue, researchers have undertaken numerous studies in Active Learning (AL). Although previous AL methods have shown remarkable outcomes for certain tasks, they tend to generate aleatoric uncertainty through a deterministic model while ignoring the epistemic uncertainty, which plays a crucial role in reflecting the cognitive abilities of the model. Furthermore, previous methods have generally computed uncertainty only in the classification dimension, leaving behind the dimension of regression. To address these challenges, we propose Hierarchical Evidence Aggregation in Two-dimensions (HEAT), which contains three modules: Evidential Heads (EH), Hierarchical Uncertainty Aggregation (HUA) and Two-round Queries Strategy (TQS). EH contains Evidential Classification Head (ECH) and Evidential Regression Head (ERH), leveraging Evidential Deep Learning (EDL) to estimate aleatoric uncertainty and epistemic uncertainty in the two dimensions, respectively. Based on these uncertainties, we use HUA to align instance uncertainty with image uncertainty and TQS to determine the key samples. Extensive experiments conducted on the Water Surface Object Detection Dataset (WSODD) demonstrate that HEAT outperforms existing state-of-the-art methods in the WSOD task.

# Environment Info
```
sys.platform: linux

Python: 3.7.13  
Pytorch : 1.10.0+cu111  
TorchVision: 0.11.0+cu111  
OpenCV: 4.7.0.72  
MMCV-full: 1.4.0  
```
# Dataset
Please click here to get the dataset: https://pan.baidu.com/s/1-xT6fwH3alW78uCsm9VjRA (the password: 1234).

# Acknowledgement
Our code is based on the implementations of [ACTIVE LEARNING FOR OBJECT DETECTION WITH EVIDENTIAL DEEP LEARNING AND HIERARCHICAL UNCERTAINTY AGGREGATION](https://github.com/MoonLab-YH/AOD_MEH_HUA).