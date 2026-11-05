"""
Training script for ML Classification Models
Implements 5 classification models and saves trained models with metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import joblib
import os
import json

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('NPHA-doctor-visits.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for missing values and handle them
print(f"\nMissing values:\n{df.isnull().sum()}")
df = df.dropna()

# Separate features and target
X = df.drop('Number of Doctors Visited', axis=1)
y = df['Number of Doctors Visited']

print(f"\nTarget distribution:\n{y.value_counts().sort_index()}")
print(f"Number of classes: {len(y.unique())}")

# Convert to binary classification (0-2 doctors = Low, 3+ = High)
y_binary = (y >= 3).astype(int)
print(f"Binary target distribution:\n{y_binary.value_counts()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')
print("Scaler saved!")

# Dictionary to store results
results = {}

# 1. Logistic Regression
print("\n" + "="*50)
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'auc': roc_auc_score(y_test, y_pred_proba_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'mcc': matthews_corrcoef(y_test, y_pred_lr)
}
joblib.dump(lr_model, 'models/logistic_regression.joblib')
print("✓ Logistic Regression trained and saved!")
print(f"Metrics: {results['Logistic Regression']}")

# 2. Decision Tree Classifier
print("\n" + "="*50)
print("Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'auc': roc_auc_score(y_test, y_pred_proba_dt),
    'precision': precision_score(y_test, y_pred_dt),
    'recall': recall_score(y_test, y_pred_dt),
    'f1': f1_score(y_test, y_pred_dt),
    'mcc': matthews_corrcoef(y_test, y_pred_dt)
}
joblib.dump(dt_model, 'models/decision_tree.joblib')
print("✓ Decision Tree trained and saved!")
print(f"Metrics: {results['Decision Tree']}")

# 3. K-Nearest Neighbor Classifier
print("\n" + "="*50)
print("Training K-Nearest Neighbor Classifier...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

results['KNN'] = {
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'auc': roc_auc_score(y_test, y_pred_proba_knn),
    'precision': precision_score(y_test, y_pred_knn),
    'recall': recall_score(y_test, y_pred_knn),
    'f1': f1_score(y_test, y_pred_knn),
    'mcc': matthews_corrcoef(y_test, y_pred_knn)
}
joblib.dump(knn_model, 'models/knn.joblib')
print("✓ KNN trained and saved!")
print(f"Metrics: {results['KNN']}")

# 4. Naive Bayes Classifier (Gaussian)
print("\n" + "="*50)
print("Training Naive Bayes Classifier...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)
y_pred_proba_nb = nb_model.predict_proba(X_test_scaled)[:, 1]

results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, y_pred_nb),
    'auc': roc_auc_score(y_test, y_pred_proba_nb),
    'precision': precision_score(y_test, y_pred_nb),
    'recall': recall_score(y_test, y_pred_nb),
    'f1': f1_score(y_test, y_pred_nb),
    'mcc': matthews_corrcoef(y_test, y_pred_nb)
}
joblib.dump(nb_model, 'models/naive_bayes.joblib')
print("✓ Naive Bayes trained and saved!")
print(f"Metrics: {results['Naive Bayes']}")

# 5. Support Vector Machine
print("\n" + "="*50)
print("Training Support Vector Machine...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'auc': roc_auc_score(y_test, y_pred_proba_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'f1': f1_score(y_test, y_pred_svm),
    'mcc': matthews_corrcoef(y_test, y_pred_svm)
}
joblib.dump(svm_model, 'models/svm.joblib')
print("✓ SVM trained and saved!")
print(f"Metrics: {results['SVM']}")

# Save results to JSON
with open('models/metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

# Display summary table
print("\n" + "="*80)
print("SUMMARY OF ALL MODELS")
print("="*80)
results_df = pd.DataFrame(results).T
print(results_df.round(4))
print("="*80)

# Save test data (first 100 samples)
test_data = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
test_data['Number of Doctors Visited'] = y_test.values  # Add binary target
test_data.to_csv('test_data.csv', index=False)
print(f"\nTest data saved! ({len(test_data)} samples)")

print("\n✓ All models trained and saved successfully!")
print("✓ Models saved in: models/")
print("✓ Test data saved as: test_data.csv")
