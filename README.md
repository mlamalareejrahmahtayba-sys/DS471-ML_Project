# Dry Bean Classification – DS471 Machine Learning Project

Classifying dry beans into seven varieties from their physical and geometric measurements, and comparing two traditional machine-learning models with a neural network.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xbr-_7Yd_i3wk7mRFZRtipG5qxKO8cDi?usp=sharing)

## Overview

Some dry bean varieties look very similar, which makes manual sorting slow and error-prone. This project uses the [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean) to build and compare three classifiers:

| Model | Type | Library |
| --- | --- | --- |
| Logistic Regression | Linear baseline | scikit-learn |
| Random Forest | Tree-based ensemble | scikit-learn |
| Multilayer Perceptron (MLP) | Neural network | PyTorch |

The full workflow covers data cleaning, exploratory data analysis, feature selection and engineering, a stratified train/validation/test split, hyperparameter tuning, training and validation analysis, and a final comparison on an untouched test set.

## Dataset

| Property | Value |
| --- | --- |
| Source | UCI Machine Learning Repository (ID 602) |
| Records | 13,611 (13,543 after removing 68 exact duplicates) |
| Input features | 16 numerical (size and shape measurements) |
| Target | `Class` – 7 varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Missing values | None |

The classes are imbalanced: DERMASON is the largest class (26.18%) and BOMBAY the smallest (3.85%). For this reason, **Macro F1** is used as the main metric for tuning and comparison.

The dataset is not included in this repository. The notebook downloads it automatically from UCI, so no manual download is needed (an internet connection is required).

## Methodology

**1. Data cleaning and EDA.** Checked missing values, duplicates, inconsistent labels and data types, and outliers (IQR method). Outliers were kept because they reflect natural variation in bean size and shape. EDA includes the class distribution, feature histograms, a correlation heatmap, scatter plots and boxplots by class.

**2. Data splitting.** Stratified 70% / 15% / 15% split (9,479 / 2,032 / 2,032 records) with `random_state=42`. The test set was used only once, for the final comparison.

**3. Feature selection and scaling.** Features were ranked by mutual information, and a feature was kept only if its absolute correlation with every already-selected feature was at most 0.95. This reduced 16 features to 9: `Perimeter`, `ShapeFactor2`, `ShapeFactor1`, `MinorAxisLength`, `ShapeFactor3`, `Roundness`, `ShapeFactor4`, `Solidity`, `Extent`. Engineered features (log-size and ratio features) were tested with 5-fold CV but gave no meaningful improvement. Selection and `StandardScaler` were fitted on training data only, and refitted inside every CV fold to avoid data leakage.

**4. Models and tuning.**

| Model | Tuning method | Selected hyperparameters |
| --- | --- | --- |
| Logistic Regression | GridSearchCV, 5-fold stratified CV (14 combinations) | `C=1`, `class_weight=None` |
| Random Forest | RandomizedSearchCV, 3-fold stratified CV (10 candidates) | `n_estimators=250`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=1`, `max_features="sqrt"`, `class_weight="balanced"` |
| MLP | Grid search on the validation set (72 configurations) | Hidden layers (64, 32), dropout 0.0, lr 0.003, weight decay 1e-4, batch size 64 |

The MLP uses Linear → BatchNorm → ReLU → Dropout blocks, the AdamW optimizer, cross-entropy loss, early stopping (patience 12) and a ReduceLROnPlateau scheduler. The final network has 3,143 trainable parameters; its best checkpoint was at epoch 23.

## Results

Final comparison on the held-out test set (2,032 records):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC (OvR) |
| --- | --- | --- | --- | --- | --- |
| **Neural Network (MLP)** | **0.9262** | **0.9367** | **0.9383** | **0.9372** | **0.9949** |
| Random Forest | 0.9193 | 0.9326 | 0.9309 | 0.9316 | 0.9945 |
| Logistic Regression | 0.9154 | 0.9280 | 0.9292 | 0.9285 | 0.9939 |

All plots, including the MLP training and validation loss curves and the final comparison chart, can be viewed directly in the notebook.

**Key findings**

- **All three models performed well and closely.** Test Macro F1 ranged from 0.9285 to 0.9372, so no model has a large practical advantage over the others.
- **Logistic Regression** reached a test Macro F1 of 0.9285 with the smallest train–validation gap (0.0081). This shows that most bean classes are already close to linearly separable after scaling, and it remains a strong choice when a simple, fast and interpretable model is preferred.
- **Random Forest** reached a test Macro F1 of 0.9316. Tuning reduced its train–validation Macro F1 gap from 0.0690 to 0.0483, but it still overfits the most of the three models. Its most important features were ShapeFactor1, ShapeFactor3, Perimeter and MinorAxisLength.
- **Neural Network (MLP)** achieved the best score on every test metric (Macro F1 0.9372), and its training and validation loss curves stay close together, so it was selected as the most suitable model.
- **Common patterns across models:** BOMBAY was classified perfectly by all three models on the validation set, while SIRA was the hardest class for all of them (validation F1 0.8671–0.8742), mostly because of confusion with DERMASON.
- **Feature selection** reduced the inputs from 16 to 9 features without lowering cross-validation performance.

## Future Work

- Try more models, such as XGBoost and SVM, to see if they can beat the current results.
- Combine the models using ensemble methods like Stacking, so the final prediction benefits from the strengths of each model.
- Improve the separation between SIRA and DERMASON, the two classes most often confused, by studying their misclassified samples.

## Repository Structure

```
├── DS471_project_notebook.ipynb   # Complete workflow, from data loading to final comparison
├── DS471_Project.pdf              # Project report
└── README.md
```

When the notebook runs, it also creates local `figures/` and `results/` folders containing saved plots, CSV result tables and the trained MLP model. These are generated outputs and are not included in the repository.

## How to Run

**Google Colab (recommended):** open the notebook with the Colab badge above and select *Runtime → Run all*. A GPU runtime speeds up the MLP grid search but is not required.

**Locally:**

```bash
git clone <repository-url>
cd <repository-folder>
pip install pandas numpy matplotlib seaborn scikit-learn torch ucimlrepo openpyxl joblib
jupyter notebook DS471_project_notebook.ipynb
```

Random seeds are fixed (`random_state=42`, `torch.manual_seed(42)`). Results were produced on a Colab GPU; running on a CPU or a different library version may change the MLP results slightly.

## Team

| Name | Student ID |
| --- | --- |
| Amal Arishi | 230040071 |
| Areej Saleem | 230010521 |
| Rahmah Muminah | 220002832 |
| Tayba Ba-Aqail | 220036240 |

**Course:** DS471 – Machine Learning (CRN 10702)

## Acknowledgements

Dataset: Koklu, M. and Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507. Available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean).
