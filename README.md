# Wine Quality Prediction

A machine learning project that predicts whether a wine is of **good quality** (quality score > 5) or **poor quality** (quality score ≤ 5) based on its physicochemical properties. The project covers the full ML pipeline: data loading, exploratory analysis, preprocessing, feature engineering, model training, and evaluation.

---

## Project Goal

The aim is to build a binary classifier that can automatically assess wine quality from measurable chemical attributes — such as acidity, residual sugar, alcohol content, and more. This kind of model could assist winemakers, distributors, or quality control teams in making faster, data-driven decisions without relying solely on human tasters.

---

## Dataset

**File:** `winequalityN.csv`

The dataset contains **6,497 wine samples** (both red and white) with **13 columns**:

| Column | Type | Description |
|---|---|---|
| `type` | object | Wine type: `"white"` or `"red"` |
| `fixed acidity` | float64 | Non-volatile acids in wine |
| `volatile acidity` | float64 | Acetic acid level (high amounts → vinegar taste) |
| `citric acid` | float64 | Adds freshness and flavour |
| `residual sugar` | float64 | Sugar remaining after fermentation |
| `chlorides` | float64 | Salt content |
| `free sulfur dioxide` | float64 | Free SO₂ in wine |
| `total sulfur dioxide` | float64 | Total SO₂ (free + bound) |
| `density` | float64 | Density of wine |
| `pH` | float64 | Acidity/alkalinity of wine |
| `sulphates` | float64 | Antimicrobial/antioxidant additive |
| `alcohol` | float64 | Percentage alcohol content |
| `quality` | int64 | Score between 3 and 9 (rated by experts) |

---

## Step-by-Step Walkthrough

### Step 1 — Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
```

All standard data science and ML libraries are imported upfront. `XGBoost` is a powerful gradient boosting framework, while `LogisticRegression` and `SVC` are classical approaches for binary classification.

---

### Step 2 — Load the Dataset

```python
df = pd.read_csv('winequalityN.csv')
print(df.head())
```

The first five rows are displayed to verify the dataset structure. Sample data shows multiple white wines, all rated quality `6`.

---

### Step 3 — Inspect Data Types and Shape

```python
df.info()
```

**Output summary:**
- `6,497 entries`, `13 columns`
- `type` is an `object` (string), needs encoding
- All other features are `float64` or `int64`

---

### Step 4 — Descriptive Statistics

```python
df.describe().T
```

A transposed summary shows count, mean, std, min, and percentiles for each feature:

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| fixed acidity | 7.22 | 1.30 | 3.80 | 15.90 |
| volatile acidity | 0.34 | 0.16 | 0.08 | 1.58 |
| alcohol | 10.49 | 1.19 | 8.00 | 14.90 |
| quality | 5.82 | 0.87 | 3.00 | 9.00 |

> **Insight:** Most wines score between 5 and 7, making this a relatively imbalanced binary classification problem.

---

### Step 5 — Handle Missing Values

```python
df.isnull().sum()
```

Several columns had missing values:

| Column | Missing Count |
|---|---|
| fixed acidity | 10 |
| volatile acidity | 8 |
| citric acid | 3 |
| residual sugar | 2 |
| chlorides | 2 |
| pH | 9 |
| sulphates | 4 |

**Imputation strategy:** Fill each missing value with the **column mean**.

```python
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
```

> **Why mean imputation?** It is fast, preserves the mean of the distribution, and works well when data is roughly normally distributed. However, it may reduce variance slightly.

---

### Step 6 — Exploratory Visualisation

#### 6a. Feature Histograms

```python
df.hist(bins=20, figsize=(10, 10))
plt.show()
```

Histograms reveal the shape and spread of each feature's distribution. Most features are right-skewed (e.g., residual sugar), which can affect linear models. The `quality` column shows that most wines cluster around scores 5–7.

#### 6b. Quality vs Alcohol Bar Chart

```python
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()
```

This chart explores the relationship between quality score and alcohol content. Higher-quality wines tend to have higher alcohol concentrations, suggesting `alcohol` is an informative feature for prediction.

---

### Step 7 — Encode the `type` Column

```python
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')
```

The string column `type` (`"white"`, `"red"`) is converted to numeric for correlation analysis. At this stage it becomes `NaN` for strings, but a direct replacement is done later (`"white"` → 1, `"red"` → 0).

---

### Step 8 — Correlation Heatmap & Feature Removal

```python
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
```

A boolean heatmap highlights feature pairs with correlation **greater than 0.7**.

**Finding:** `free sulfur dioxide` and `total sulfur dioxide` are highly correlated (r > 0.7).

**Action:** Remove `total sulfur dioxide` to avoid multicollinearity.

```python
df = df.drop('total sulfur dioxide', axis=1)
```

> **Why remove correlated features?** Including redundant features doesn't add new information. It can confuse certain models (especially linear ones), inflate coefficients, and slow training. This is especially important for Logistic Regression and SVC.

---

### Step 9 — Create Binary Target Variable

```python
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
```

Rather than predicting the exact quality score (a hard regression problem), the target is simplified into a binary label:

- **1** → Wine quality > 5 (good)
- **0** → Wine quality ≤ 5 (poor)

This makes the problem more tractable and practically useful.

---

### Step 10 — Encode Wine Type

```python
df.replace({'white': 1, 'red': 0}, inplace=True)
```

The `type` column is encoded as a binary numeric feature.

---

### Step 11 — Train-Test Split

```python
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40
)
```

An **80:20** split is used:
- **Training set:** 5,197 samples
- **Test set:** 1,300 samples

`random_state=40` ensures reproducible results.

After splitting, missing values (if any remained) are handled via `SimpleImputer(strategy='mean')` — fitting on training data only to prevent data leakage.

---

### Step 12 — Feature Normalisation

```python
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)
```

**MinMaxScaler** scales each feature to the range **[0, 1]**:

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

> **Why normalise?** Distance-based models like SVC and gradient-sensitive models like Logistic Regression benefit greatly from features on a common scale. Without scaling, features with large ranges (e.g., `total sulfur dioxide`) would dominate those with small ranges (e.g., `pH`).

> **Important:** The scaler is **fit only on training data** and then applied to test data. This prevents data leakage — the model should never "see" test statistics during training.

---

### Step 13 — Model Training

Three models are trained and evaluated using **ROC-AUC score**:

```python
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain, ytrain)
    print('Training Accuracy:', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy:', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
```

**Results:**

| Model | Train AUC | Validation AUC |
|---|---|---|
| Logistic Regression | 0.697 | 0.687 |
| **XGBoost Classifier** | **0.975** | **0.798** |
| SVC (RBF kernel) | 0.720 | 0.707 |

> **ROC-AUC** measures the model's ability to distinguish between classes across all thresholds. A score of 1.0 is perfect; 0.5 is random guessing.

#### Model Descriptions

- **Logistic Regression:** A linear classifier that models the probability of a binary outcome using a sigmoid function. Fast and interpretable, but limited in capturing non-linear relationships. Its similar train/test AUC suggests it's neither overfitting nor underfitting — just limited in expressiveness.

- **XGBoost (Extreme Gradient Boosting):** An ensemble method that builds decision trees sequentially, each correcting the errors of the previous. The large gap between training AUC (0.975) and validation AUC (0.798) indicates **some overfitting**, but it still generalises best among the three.

- **SVC (Support Vector Classifier with RBF kernel):** Finds the optimal hyperplane separating classes in a high-dimensional space using the Radial Basis Function kernel. A middle ground between LR and XGBoost in both performance and complexity.

**Winner: XGBClassifier** with the highest validation AUC of **0.798**.

---

### Step 14 — Confusion Matrix

```python
cm = confusion_matrix(ytest, models[1].predict(xtest))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=models[1].classes_)
disp.plot()
plt.show()
```

The confusion matrix for XGBoost on the test set breaks down predictions:

- **True Positives (TP):** Good wines correctly identified as good
- **True Negatives (TN):** Poor wines correctly identified as poor
- **False Positives (FP):** Poor wines misclassified as good
- **False Negatives (FN):** Good wines misclassified as poor

This visual aids in understanding where the model makes mistakes.


### Step 15 — Classification Report

   python
print(metrics.classification_report(ytest, models[1].predict(xtest)))


**Output (XGBoost):**

```
              precision    recall  f1-score   support

           0       0.76      0.72      0.74       474
           1       0.85      0.87      0.86       826

    accuracy                           0.82      1300
   macro avg       0.80      0.80      0.80      1300
weighted avg       0.82      0.82      0.82      1300
```

| Metric | Class 0 (Poor) | Class 1 (Good) |

| Precision | 0.76 | 0.85 |
| Recall | 0.72 | 0.87 |
| F1-Score | 0.74 | 0.86 |
| Support | 474 | 826 |

**Overall Accuracy: 82%**

> - **Precision:** Of wines predicted as good, 85% actually are.
> - **Recall:** Of all actually good wines, 87% are correctly identified.
> - **F1-Score:** The harmonic mean of precision and recall — balances both metrics.
> - The model performs better on **Class 1 (good wines)**, likely because they form the majority class (826 vs 474 samples).

---

## Conclusion

The project successfully builds a binary wine quality classifier using physicochemical features. Key findings:

1. **XGBoost** achieves the best validation performance with **82% accuracy** and **0.798 AUC**, demonstrating the power of ensemble tree-based methods on tabular data.
2. **Feature engineering** — converting quality into a binary label, encoding wine type, and removing the correlated `total sulfur dioxide` — significantly simplifies the problem while retaining predictive signal.
3. **Alcohol content** shows a positive relationship with wine quality, suggesting it is one of the more important features.
4. The model generalises reasonably well (train AUC 0.975 → test AUC 0.798), though there is room to reduce overfitting via hyperparameter tuning, regularisation, or cross-validation.

### Potential Next Steps

- **Hyperparameter tuning** (e.g., GridSearchCV or Optuna) to improve XGBoost generalisation
- **SHAP values** for feature importance and model explainability
- **Class imbalance handling** (SMOTE or class weights) since poor-quality wines are underrepresented
- **Cross-validation** for more robust model evaluation
- **Try other models** (Random Forest, LightGBM, Neural Networks) for comparison



## Requirements


numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost


Install with:

   bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost





