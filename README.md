# Diffusion-powered Data Augmentation and Explainable Boosting Ensemble Learning for Cyberattack Detection in Industrial Networks

This repository contains the implementation and datasets used in our paper:

**"Diffusion-powered Data Augmentation and Explainable Boosting Ensemble Learning for Cyberattack Detection in Industrial Networks"**  
_Authors:Tuyen T. Nguyen, Phong H. Nguyen, Hanh P. Du, Hoa N. Nguyen 

> Please cite our work if you use this code or data in your research.

---

## 📌 Abstract
Detecting cyberattacks on Industrial Control Systems, particularly those using the IEC 60870-5-104 protocol, is critical and increasingly challenging in Industry 4.0. This research introduces DABEL, a novel framework integrating  method that effectively addresses these challenges through diffusion-powered data augmentation and explainable boosting ensemble learning. DABEL tackles the problem in four phases: First, it leverages Shapley Additive Explanations  to pinpoint the most relevant features for attack detection, enhancing interpretability and reducing noise. Second, DABEL generates realistic synthetic attack samples using advanced generative models, including Diffusion Models,  significantly enhancing the diversity and robustness of the training data  to address severe class imbalance in ICS datasets. Third, it employs a weighted voting ensemble learning strategy that integrates multiple powerful AI models concurrently, with parallel boosting and inverse-error weighting,  to improve performance and resilience against adversarial attacks. Finally, through comprehensive experiments on the IEC 60870-5-104 dataset, DABEL demonstrates superior accuracy compared to existing cutting-edge methods, achieving a high accuracy of 86.83%, AUC ROC score of 98.92%, and an exceptionally low false alarm rate of 1.18%.

## 📜 Citation

If you find our work useful, please cite:

@article{jDABEL25,
title={Diffusion-powered Data Augmentation and Explainable Boosting Ensemble Learning for Cyberattack Detection in Industrial Networks},
author={Tuyen T. Nguyen, Phong H. Nguyen, Hanh P. Du, Hoa N. Nguyen },
journal={Intelligent Automation & Soft Computing},
publisher={Tech Science Press},
note={submitted},
year={2025}
}
