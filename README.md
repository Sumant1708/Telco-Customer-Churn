# Telco Customer Churn Prediction

A machine learning project that explores customer churn prediction using the Telco Customer Churn dataset. The project focuses not only on classification performance, but also on choosing suitable evaluation metrics, comparing models with stratified cross-validation, examining decision thresholds, and evaluating predicted-probability calibration.

## Project Objectives

- Explore customer attributes and their relationship with churn.
- Prepare numerical and categorical features for machine learning.
- Compare Logistic Regression, Support Vector Classifier (SVC), and K-Nearest Neighbors (KNN).
- Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Investigate how changing the classification threshold affects precision and recall.
- Examine probability calibration using reliability diagrams and the Brier score.
- Study the effect of feature selection and dimensionality on KNN performance.

## Dataset

The notebook uses the Telco Customer Churn dataset (`Telco-Customer-Churn.csv`), containing **7,043 customer records and 21 columns**. Features describe customer demographics, account information, subscribed services, billing, and churn status.

The target variable is `Churn` (`Yes` / `No`). The `customerID` column is removed before model training.


## Workflow

1. **Data inspection and cleaning**
   - Convert `TotalCharges` to numeric values.
   - Handle the 11 missing `TotalCharges` values by filling them with `0`.
   - Remove `customerID` and inspect duplicate records and feature distributions.

2. **Exploratory Data Analysis**
   - Analyze churn across demographic, service, contract, and payment attributes.
   - Visualize categorical distributions, numerical relationships, and feature correlations.

3. **Preprocessing**
   - Encode the target as binary values (`Yes` = 1, `No` = 0).
   - Split the data into training and test sets using an 80/20 stratified split (`random_state=42`).
   - Apply one-hot encoding to categorical features and standardize the resulting feature matrix.

4. **Model training and evaluation**
   - Train Logistic Regression, RBF-kernel SVC, and KNN classifiers.
   - Compare test-set metrics and ROC curves.
   - Use 5-fold `StratifiedKFold` cross-validation with ROC-AUC scoring.

5. **Threshold and calibration analysis**
   - Compare precision, recall, and F1-score at probability thresholds from 0.3 to 0.7.
   - Plot reliability diagrams and calculate the Brier score.
   - Apply sigmoid calibration to Logistic Regression and compare Brier scores before and after calibration.

6. **KNN feature-selection experiment**
   - Evaluate KNN across different neighbor counts.

## Results

### Test-set model comparison

| Model | Accuracy | ROC-AUC | Churn precision | Churn recall | Churn F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8070 | 0.8418 | 0.66 | 0.57 | 0.61 |
| SVC (RBF, class_weight=balanced) | 0.7516 | 0.8168 | 0.52 | 0.77 | 0.62 |
| KNN (5 neighbors) | 0.7488 | 0.7701 | 0.53 | 0.52 | 0.52 |

### 5-fold cross-validation (ROC-AUC)

| Model | Mean ROC-AUC | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.8463 | 0.0127 |
| SVC | 0.8266 | 0.0099 |
| KNN (5 neighbors) | 0.7736 | 0.0065 |

Logistic Regression achieved the highest mean cross-validation ROC-AUC among the three baseline models. SVC produced higher churn recall on the held-out test set, illustrating the trade-off between identifying more customers likely to churn and generating more false positives.

### Threshold analysis — Logistic Regression

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.3 | 0.518 | 0.754 | 0.614 |
| 0.4 | 0.566 | 0.666 | 0.612 |
| 0.5 | 0.658 | 0.567 | 0.609 |
| 0.6 | 0.713 | 0.398 | 0.511 |
| 0.7 | 0.730 | 0.174 | 0.281 |

A lower threshold increases churn recall while reducing precision. The appropriate threshold depends on the business cost of missed churners versus unnecessary retention outreach; this project does not claim a universally optimal business threshold.

### Probability calibration

For Logistic Regression, the notebook reports:

- Brier score before sigmoid calibration: **0.138205**
- Brier score after sigmoid calibration: **0.138136**

The change is small, so calibration did not materially improve the reported Brier score in this experiment.

### KNN feature-selection experiment

The notebook evaluates KNN with different feature counts using 5-fold stratified cross-validation. The highest mean ROC-AUC in that experiment was **0.7869 with 10 selected features**, compared with **0.7736 using all 45 features** (KNN with 5 neighbors).

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Run Locally

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Place `Telco-Customer-Churn.csv` at the path expected by the notebook, or edit the `pd.read_csv()` path.
4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open and run the Telco Customer Churn notebook from top to bottom.

## Key Takeaway

This project demonstrates that model evaluation goes beyond accuracy. ROC-AUC supports ranking comparisons, precision and recall expose classification trade-offs, threshold analysis reflects different decision costs, and calibration analysis checks whether predicted probabilities align with observed outcomes.

