# Breast Cancer Diagnosis Classification

Machine learning (ML) classifier for breast cancer diagnosis using cell morphology features from the Wisconsin Diagnostic Breast Cancer dataset. Compares Logistic Regression, Random Forest, SVM, and KNN, achieving 98.8% test accuracy with Logistic Regression (AUC = 0.998).

Built as part of the Teens in Health AI and Bioinformatics summer cohort, to support a research article submitted for publication in the Teens in Health journal.

## Dataset

[Wisconsin Diagnostic Breast Cancer (WDBC) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — 569 samples, 30 numeric features describing nuclear size, shape, and texture, extracted from digitized fine needle aspirate (FNA) biopsy images. Accessed via scikit-learn's `load_breast_cancer()`.

## Methods

- Train/test split (70/30, stratified) with feature standardization
- Four classifiers compared: Logistic Regression, Random Forest, SVM (RBF), KNN
- 5-fold cross-validation, evaluated on accuracy, precision, recall, F1, and ROC-AUC
- Feature importance analysis (Random Forest) and PCA dimensionality reduction for visualization

## Results

| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9824 | 0.9883 | 0.9907 | 0.9907 | 0.9907 | 0.9981 |
| SVM (RBF) | 0.9723 | 0.9766 | 0.9813 | 0.9813 | 0.9813 | 0.9978 |
| KNN (k=5) | 0.9674 | 0.9591 | 0.9386 | 1.0000 | 0.9683 | 0.9827 |
| Random Forest | 0.9623 | 0.9357 | 0.9444 | 0.9533 | 0.9488 | 0.9913 |

**Logistic Regression performed best** overall (98.83% test accuracy, AUC = 0.998), with only 2 total misclassifications out of 171 test cases. **KNN achieved perfect recall**, catching every malignant case in the test set, at the cost of more false positives — illustrating the precision/recall trade-off relevant to diagnostic screening.

The most predictive features (via Random Forest) were related to nuclear size and shape irregularity — *worst concave points*, *worst area*, and *worst perimeter* — consistent with known pathology of malignant cell morphology. PCA showed malignant and benign cases form largely separable clusters in just 2 dimensions, capturing 63.24% of total variance.

## Tech Stack

Python, scikit-learn, pandas, NumPy, matplotlib, seaborn

## Limitations

- **Small, single-source dataset.** With only 569 samples from one institution, results may not generalize well to broader or more diverse patient populations.
- **Pre-extracted features, not raw images.** This pipeline classifies based on numbers that were already computed from microscope images — it doesn't address the harder real-world task of extracting those features from raw image data.
- **No external validation set.** Performance was measured on a held-out test split from the same dataset, not on independent data from a different source, so reported accuracy may be somewhat optimistic.
- **Binary classification only.** Real diagnosis involves more nuance than malignant/benign (e.g., cancer subtype, stage), which this model doesn't address.
- **No clinical/demographic context.** The model uses only cell morphology; it doesn't incorporate patient age, family history, or genetic risk factors that a real diagnostic pipeline would typically consider.
- **Proof-of-concept, not clinical tool.** This project demonstrates that AI can detect meaningful patterns in diagnostic data; it has not been validated for real-world clinical use.

## Future Work

- Validate the model on an independent, external breast cancer dataset to test generalizability beyond this single source.
- Extend the pipeline to work directly on raw histopathology or mammography images using convolutional neural networks (CNNs), rather than pre-extracted features.
- Incorporate multi-omics data (e.g., gene expression alongside morphology) to test whether combining data types improves diagnostic accuracy.
- Explore model interpretability methods (e.g., SHAP values) to better explain individual predictions in a way that's useful to clinicians.
- Test performance specifically on borderline/ambiguous cases, where diagnostic support tools matter most in practice.

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python3 breast_cancer_analysis.py
```

Results (plots and CSVs) will be saved to an `outputs/` folder.