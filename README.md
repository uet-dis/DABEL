# Diffusion-based Augmentation and Explainable Ensemble Learning for Cyberattack Detection

This repository contains the implementation and datasets used in our paper:

**"DABEL: Diffusion-based Augmentation and Explainable Ensemble Learning for Cyberattack Detection in IEC 60870-5-104 Networks"**  
_Author(s):Tuyen T. Nguyen, Phong H. Nguyen, Hanh P. Du, Hoa N. Nguyen 
_Journal: International Journal of Intelligent Engineering and Systems (IJIES), 2026

> Please cite our work if you use this code or data in your research.

---

## 📌 Abstract
Cyberattack detection in IEC 60870-5-104 Industrial Control System networks remains challenging due to high-dimensional traffic features, limited interpretability, and evolving attack patterns. This paper presents DABEL, an explainable intrusion detection framework that combines SHAP-guided feature reduction, diffusion-based data augmentation, and weighted ensemble learning for IEC 60870-5-104 cyberattack detection. SHAP-based feature selection is first employed to identify informative traffic features and reduce data dimensionality. Next, diffusion-based augmentation methods, including Forest Diffusion Models (FDM) and Conditional Flow Matching (CFM), are used to generate synthetic attack samples to enrich attack-pattern diversity and improve model generalization. Finally, a weighted ensemble integrating XGBoost, RandomForest, and ExtraTrees is developed to improve detection stability and inference efficiency through parallel execution. Experiments on the IEC 60870-5-104 dataset show that DABEL achieves 86.83% accuracy, 86.39% macro F1-score, 98.92% AUC, and a 1.18% false alarm rate (FAR). The results indicate that the integration of SHAP-guided feature reduction, diffusion-based augmentation, and weighted ensemble learning improves cyberattack detection performance in IEC 60870-5-104 networks.

## 📜 Citation

If you find our work useful, please cite:

@article{jDABEL26,
title={DABEL: Diffusion-based Augmentation and Explainable Ensemble Learning for Cyberattack Detection in IEC 60870-5-104 Networks},
author={Tuyen T. Nguyen, Phong H. Nguyen, Hanh P. Du, Hoa N. Nguyen },
journal={International Journal of Intelligent Engineering and Systems (IJIES)},
note={in revision},
year={2026}
}
