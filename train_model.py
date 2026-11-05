"""
ML Classification Models Training Script
Trains 5 classification models on NPHA doctor visits dataset
Saves trained models and evaluation metrics
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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib
import os
from pathlib import Path

# Create model directory
model_dir = Path('model')
model_dir.mkdir(exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('NPHA-doctor-visits.csv')

# Display dataset info
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget variable value counts:")
print(df['Number of Doctors Visited'].value_counts().sort_index())

# Prepare features and target
X = df.drop('Number of Doctors Visited', axis=1)
y = df['Number of Doctors Visited']

# Convert to classification problem (multi-class)
print(f"\nNumber of classes: {len(y.unique())}")
print(f"Class distribution: {sorted(y.unique())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, model_dir / 'scaler.pkl')
print("\nScaler saved.")

# Dictionary to store models and results
models = {}
results = {}

# 1. Logistic Regression
print("\n" + "="*60)
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr_model

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)

# Metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr, multi_class='ovr')
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
mcc_lr = matthews_corrcoef(y_test, y_pred_lr)

results['Logistic Regression'] = {
    'Accuracy': accuracy_lr,
    'AUC': auc_lr,
    'Precision': precision_lr,
    'Recall': recall_lr,
    'F1': f1_lr,
    'MCC': mcc_lr,
    'y_pred': y_pred_lr,
    'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
    'classification_report': classification_report(y_test, y_pred_lr)
}

print(f"Accuracy: {accuracy_lr:.4f} | AUC: {auc_lr:.4f} | Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f} | F1: {f1_lr:.4f} | MCC: {mcc_lr:.4f}")

# 2. Decision Tree Classifier
print("\n" + "="*60)
print("Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=5)
dt_model.fit(X_train_scaled, y_train)
models['Decision Tree'] = dt_model

y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt, multi_class='ovr')
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
mcc_dt = matthews_corrcoef(y_test, y_pred_dt)

results['Decision Tree'] = {
    'Accuracy': accuracy_dt,
    'AUC': auc_dt,
    'Precision': precision_dt,
    'Recall': recall_dt,
    'F1': f1_dt,
    'MCC': mcc_dt,
    'y_pred': y_pred_dt,
    'confusion_matrix': confusion_matrix(y_test, y_pred_dt),
    'classification_report': classification_report(y_test, y_pred_dt)
}

print(f"Accuracy: {accuracy_dt:.4f} | AUC: {auc_dt:.4f} | Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f} | F1: {f1_dt:.4f} | MCC: {mcc_dt:.4f}")

# 3. K-Nearest Neighbors
print("\n" + "="*60)
print("Training K-Nearest Neighbors Classifier...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
models['KNN'] = knn_model

y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, y_pred_proba_knn, multi_class='ovr')
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
mcc_knn = matthews_corrcoef(y_test, y_pred_knn)

results['KNN'] = {
    'Accuracy': accuracy_knn,
    'AUC': auc_knn,
    'Precision': precision_knn,
    'Recall': recall_knn,
    'F1': f1_knn,
    'MCC': mcc_knn,
    'y_pred': y_pred_knn,
    'confusion_matrix': confusion_matrix(y_test, y_pred_knn),
    'classification_report': classification_report(y_test, y_pred_knn)
}

print(f"Accuracy: {accuracy_knn:.4f} | AUC: {auc_knn:.4f} | Precision: {precision_knn:.4f}")
print(f"Recall: {recall_knn:.4f} | F1: {f1_knn:.4f} | MCC: {mcc_knn:.4f}")

# 4. Naive Bayes (Gaussian)
print("\n" + "="*60)
print("Training Naive Bayes Classifier...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
models['Naive Bayes'] = nb_model

y_pred_nb = nb_model.predict(X_test_scaled)
y_pred_proba_nb = nb_model.predict_proba(X_test_scaled)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
auc_nb = roc_auc_score(y_test, y_pred_proba_nb, multi_class='ovr')
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')
mcc_nb = matthews_corrcoef(y_test, y_pred_nb)

results['Naive Bayes'] = {
    'Accuracy': accuracy_nb,
    'AUC': auc_nb,
    'Precision': precision_nb,
    'Recall': recall_nb,
    'F1': f1_nb,
    'MCC': mcc_nb,
    'y_pred': y_pred_nb,
    'confusion_matrix': confusion_matrix(y_test, y_pred_nb),
    'classification_report': classification_report(y_test, y_pred_nb)
}

print(f"Accuracy: {accuracy_nb:.4f} | AUC: {auc_nb:.4f} | Precision: {precision_nb:.4f}")
print(f"Recall: {recall_nb:.4f} | F1: {f1_nb:.4f} | MCC: {mcc_nb:.4f}")

# 5. Support Vector Machine
print("\n" + "="*60)
print("Training Support Vector Machine...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
models['SVM'] = svm_model

y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_pred_proba_svm, multi_class='ovr')
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
mcc_svm = matthews_corrcoef(y_test, y_pred_svm)

results['SVM'] = {
    'Accuracy': accuracy_svm,
    'AUC': auc_svm,
    'Precision': precision_svm,
    'Recall': recall_svm,
    'F1': f1_svm,
    'MCC': mcc_svm,
    'y_pred': y_pred_svm,
    'confusion_matrix': confusion_matrix(y_test, y_pred_svm),
    'classification_report': classification_report(y_test, y_pred_svm)
}

print(f"Accuracy: {accuracy_svm:.4f} | AUC: {auc_svm:.4f} | Precision: {precision_svm:.4f}")
print(f"Recall: {recall_svm:.4f} | F1: {f1_svm:.4f} | MCC: {mcc_svm:.4f}")

# Save all models
print("\n" + "="*60)
print("Saving trained models...")
for model_name, model in models.items():
    model_path = model_dir / f"{model_name.replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved: {model_path}")

# Save results summary
print("\nSaving results summary...")
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['Accuracy'] for m in results.keys()],
    'AUC': [results[m]['AUC'] for m in results.keys()],
    'Precision': [results[m]['Precision'] for m in results.keys()],
    'Recall': [results[m]['Recall'] for m in results.keys()],
    'F1': [results[m]['F1'] for m in results.keys()],
    'MCC': [results[m]['MCC'] for m in results.keys()]
})

results_df.to_csv('model_results.csv', index=False)
print("Results saved to model_results.csv")

# Print summary table
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

# Save test data for deployment
print("\n" + "="*60)
print("Saving test data for Streamlit app...")
test_data = X_test.copy()
test_data['Number of Doctors Visited'] = y_test.values
test_data.to_csv('test_data.csv', index=False)
print(f"Test data saved to test_data.csv ({len(test_data)} samples)")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
