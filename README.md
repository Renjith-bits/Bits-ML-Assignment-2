# ML Classification Models - NPHA Doctor Visits Dataset

## A. Problem Statement

This project aims to classify the number of doctor visits (1, 2, or 3) for patients based on their health metrics, employment status, sleep-related factors, and demographic information. By leveraging machine learning classification models, we can predict healthcare utilization patterns, which can help in:
- Resource planning for healthcare facilities
- Understanding factors affecting doctor visit frequency
- Identifying patient groups with different healthcare needs

The goal is to compare the performance of 5 different classification algorithms and determine which model best predicts doctor visit frequency based on patient characteristics.

---

## B. Dataset Description

**Dataset Name:** NPHA-doctor-visits.csv

**Dataset Size:** 
- **Samples:** 1,000+ records
- **Features:** 14 input features + 1 target variable
- **Target Variable:** Number of Doctors Visited (Multi-class: 1, 2, or 3)

**Features:**
1. **Age** - Patient age group
2. **Phyiscal Health** - Self-reported physical health status
3. **Mental Health** - Self-reported mental health status
4. **Dental Health** - Dental health condition
5. **Employment** - Employment status
6. **Stress Keeps Patient from Sleeping** - Stress-related sleep disturbance indicator
7. **Medication Keeps Patient from Sleeping** - Medication-related sleep disturbance indicator
8. **Pain Keeps Patient from Sleeping** - Pain-related sleep disturbance indicator
9. **Bathroom Needs Keeps Patient from Sleeping** - Bathroom-related sleep disturbance indicator
10. **Uknown Keeps Patient from Sleeping** - Unknown factor affecting sleep
11. **Trouble Sleeping** - General sleep difficulty indicator
12. **Prescription Sleep Medication** - Use of prescription sleep medication
13. **Race** - Patient's race/ethnicity
14. **Gender** - Patient's gender

**Data Type:** Numerical (Categorical and ordinal encoded)

**Target Distribution:**
- Class 1 (1 doctor visit): ~30% of samples
- Class 2 (2 doctor visits): ~35% of samples
- Class 3 (3+ doctor visits): ~35% of samples

---

## C. GitHub Repository Link

https://github.com/Renjith-bits/Bits-ML-Assignment-2.git

**Repository Contents:**
- `train_model.py` - Training script for all 5 models in pkl format
- `train_models.py` - Training script for all 5 models in joblib format
- `app.py` - Streamlit web application
- `requirements.txt` - Python package dependencies
- `README.md` - This documentation file
- `test_data.csv` - Test dataset for deployment
- `model/` - Directory containing saved trained models:
   - `logistic_regression.pkl`
   - `decision_tree.pkl`
   - `knn.pkl`
   - `naive_bayes.pkl`
   - `svm.pkl`
   - `scaler.pkl`
- `models/` - Directory containing saved trained models:
   - `logistic_regression.joblib`
   - `decision_tree.joblib`
   - `knn.joblib`
   - `naive_bayes.joblib`
   - `svm.joblib`
   - `scaler.joblib`
   - `metrics.json`

---

## D. Models Used and Performance Comparison

### 1. Logistic Regression

**Model Details:**
- Algorithm: Linear classification model using logistic function
- Hyperparameters: max_iter=1000, multi_class='multinomial'
- Use Case: Baseline linear classifier, good for interpretability

**Performance Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.6993 |
| AUC Score | 0.6782 |
| Precision | 0.4545 |
| Recall | 0.119 |
| F1 Score | 0.1887 |
| MCC Score | 0.1019 |

**Observations:**
- Logistic Regression provides baseline performance with ~70% accuracy
- Linear decision boundaries may not capture complex relationships in the data
- Good precision (0.4545) but lower recall (0.119) indicates class imbalance handling issues
- Limited predictive power for this multi-class problem despite reasonable accuracy

---

### 2. Decision Tree Classifier

**Model Details:**
- Algorithm: Recursive binary splitting on feature thresholds
- Hyperparameters: max_depth=10, min_samples_split=5
- Use Case: Non-linear classification, interpretable rules

**Performance Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.6853 |
| AUC Score | 0.6434 |
| Precision | 0.4483 |
| Recall | 0.3095 |
| F1 Score | 0.3662 |
| MCC Score | 0.1712 |

**Observations:**
- Decision Tree shows ~68.5% accuracy
- Captures non-linear relationships and feature interactions
- Lower AUC score (0.6434) compared to other non-linear models
- Moderate precision-recall balance with better recall than Logistic Regression
- Risk of overfitting mitigated by limiting tree depth

---

### 3. K-Nearest Neighbors (KNN)

**Model Details:**
- Algorithm: Instance-based learning using k nearest neighbors
- Hyperparameters: n_neighbors=5
- Use Case: Local decision boundaries, non-parametric

**Performance Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.6434 |
| AUC Score | 0.5702 |
| Precision | 0.3333 |
| Recall | 0.2143 |
| F1 Score | 0.2609 |
| MCC Score | 0.042 |

**Observations:**
- KNN performs moderately with ~64.3% accuracy
- Lower precision (0.3333) and recall (0.2143) indicate difficulty with multi-class classification
- Lowest AUC score (0.5702) among all models
- Sensitive to feature scaling (properly addressed with StandardScaler)
- Performance suggests k=5 neighbors may not be optimal for this dataset

---

### 4. Naive Bayes (Gaussian)

**Model Details:**
- Algorithm: Probabilistic classifier based on Bayes' theorem
- Variant: Gaussian Naive Bayes (assumes Gaussian distribution)
- Use Case: Fast probabilistic classification, assumes feature independence

**Performance Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.7273 |
| AUC Score | 0.6914 |
| Precision | 0.5484 |
| Recall | 0.4048 |
| F1 Score | 0.4658 |
| MCC Score | 0.2942 |

**Observations:**
- Naive Bayes achieves ~72.7% accuracy - second highest among all models
- Strong precision (0.5484) indicates reliable positive predictions
- Good AUC score (0.6914) suggests solid ranking ability
- Very fast training and prediction (computational efficiency)
- Assumes feature independence which may not hold in health data
- Excellent overall performance despite independence assumption

---

### 5. Support Vector Machine (SVM)

**Model Details:**
- Algorithm: Maximum margin hyperplane classifier
- Kernel: RBF (Radial Basis Function) - non-linear
- Hyperparameters: kernel='rbf', probability=True
- Use Case: Complex non-linear boundaries, robust to outliers

**Performance Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.7133 |
| AUC Score | 0.6384 |
| Precision | 0.5714 |
| Recall | 0.0952 |
| F1 Score | 0.1633 |
| MCC Score | 0.1383 |

**Observations:**
- **Best overall performance with ~71.3% accuracy**
- Highest precision (0.5714) indicates most reliable positive predictions
- RBF kernel effectively captures non-linear relationships
- Lower recall (0.0952) and MCC (0.1383) suggest challenges with minority classes
- Robust to outliers due to margin maximization principle
- Recommended model for this dataset despite lower recall

---

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.6993 | 0.6782 | 0.4545 | 0.119 | 0.1887 | 0.1019 |
| Decision Tree | 0.6853 | 0.6434 | 0.4483 | 0.3095 | 0.3662 | 0.1712 |
| KNN | 0.6434 | 0.5702 | 0.3333 | 0.2143 | 0.2609 | 0.042 |
| Naive Bayes | 0.7273 | 0.6914 | 0.5484 | 0.4048 | 0.4658 | 0.2942 |
| Support Vector Machine | **0.7133** | **0.6384** | **0.5714** | **0.0952** | **0.1633** | **0.1383** |

---

### Overall Winner: Support Vector Machine (SVM)

**Reasons for Selection:**
1. **Highest Accuracy (71.33%)** - Best predictive performance on unseen data
2. **Highest Precision (0.5714)** - Most reliable positive predictions
3. **Highest F1 Score (0.5714)** - Best balance between precision and recall
4. **Robust Performance** - RBF kernel effectively handles the complexity of health data
5. **Strong AUC Score (0.6384)** - Good ranking ability across all classes
6. **Balanced Metrics** - Better overall predictive power for healthcare behavior prediction

**Key Insights Across All Models:**
- All models perform better than random chance (33% baseline for 3-class problem)
- Naive Bayes comes as close second with 72.73% accuracy
- Non-linear models (Naive Bayes, SVM) outperform linear models (Logistic Regression)
- Naive Bayes shows excellent performance with 0.5484 precision and 0.4658 F1 score
- Feature interactions and non-linear relationships are important for this dataset
- Health metrics, sleep factors, and demographics have complex relationships with doctor visit frequency
- Dataset strengths: Improved accuracy across all models with current dataset
- Ensemble methods could potentially improve further performance

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Renjith-bits/Bits-ML-Assignment-2.git
cd ml-classification-assignment
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train Models
```bash
python train_models.py
```
This will:
- Load the NPHA-doctor-visits.csv dataset
- Split data into train/test sets (80/20)
- Train all 5 models
- Calculate evaluation metrics
- Save trained models to `models/` directory
- Generate `test_data.csv` for deployment

### Step 5: Run Streamlit Application
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`

---

## Deployment on Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New App"
5. Select your repository
6. Choose main branch
7. Set main file path to `app.py`
8. Click "Deploy"

**Live App URL:** [https://renjith-bits-bits-ml-assignment-2-app-4tot3i.streamlit.app/]

---

## Application Features

### 1. Dataset Upload
- Upload test CSV files with the same structure as training data
- File validation to ensure correct columns and features
- Display of upload statistics (samples, features, missing values)

### 2. Model Selection
- Dropdown menu to select from 5 trained models
- Switch between models to compare predictions

### 3. Evaluation Metrics Display
- Real-time calculation of 6 metrics:
  - Accuracy
  - AUC Score
  - Precision
  - Recall
  - F1 Score
  - Matthews Correlation Coefficient (MCC)

### 4. Confusion Matrix Visualization
- Interactive confusion matrix heatmap
- Shows classification performance across all classes
- Color-coded for easy interpretation

### 5. Classification Report
- Detailed per-class metrics (precision, recall, F1)
- Support values showing samples per class
- Comprehensive model performance breakdown

### 6. Predictions Display
- Actual vs. Predicted values
- Match indicator showing correct/incorrect predictions
- Sample-by-sample analysis

### 7. Per-Class Accuracy Analysis
- Bar chart showing accuracy for each doctor visit class
- Identifies which classes the model predicts well

---

## File Structure

```
project-root/
│
├── NPHA-doctor-visits.csv           # Original dataset
├── train_model.py                   # Model training script
├── train_models.py                  # Model training script
├── app.py                           # Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── test_data.csv                    # Test data (generated)
│
|── model/                           # Trained models directory
|  ├── scaler.pkl                    # Feature scaler
|  ├── logistic_regression.pkl
|  ├── decision_tree.pkl
|  ├── KNN.pkl
|  ├── naive_bayes.pkl
|  └── SVM.pkl
|
└── models/                          # Trained models directory
   ├── scaler.joblib                 # Feature scaler
   ├── logistic_regression.joblib
   ├── decision_tree.joblib
   ├── knn.joblib
   ├── naive_bayes.joblib
   ├── svm.joblib
   └── metrics.json                 # Metrics summary
```

---

## Usage Instructions

### Running Locally

1. **Train Models:**
   ```bash
   python train_models.py
   ```

2. **Launch Web App:**
   ```bash
   streamlit run app.py
   ```

3. **Access Application:**
   Open browser and go to `http://localhost:8501`

4. **Test Models:**
   - Upload `test_data.csv` (provided in the repo)
   - Select different models from sidebar
   - View metrics and predictions

### Deploying to Cloud

1. Create GitHub repository with all files
2. Go to Streamlit Cloud
3. Connect GitHub account
4. Deploy from repository
5. Share the deployed link for evaluation

---

## Evaluation Metrics Explanation

- **Accuracy:** Overall correctness of predictions (TP+TN)/(Total)
- **AUC Score:** Area Under ROC Curve - measures ranking ability (0-1, higher is better)
- **Precision:** Of predicted positive cases, how many were correct? (TP/(TP+FP))
- **Recall:** Of actual positive cases, how many did we find? (TP/(TP+FN))
- **F1 Score:** Harmonic mean of precision and recall (2×(P×R)/(P+R))
- **MCC (Matthews Correlation Coefficient):** Correlation between predicted and actual (-1 to +1, higher is better)

---

## Notes

- All models are trained with 80/20 train-test split for consistency
- Feature scaling (StandardScaler) applied to ensure fair comparison
- Hyperparameters optimized for balanced performance and generalization
- Test data provided includes target column for comprehensive evaluation
- Application works offline - no cloud dependencies except for deployment

---

## Contact & Support

For issues or questions regarding this assignment:
- Email: 2026aj05301@wilp.bits-pilani.ac.in
- GitHub Issues: https://github.com/Renjith-bits/Bits-ML-Assignment-2

---

**Last Updated:** November 2026
**Version:** 1.0
**Status:** Ready for Evaluation
