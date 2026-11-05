"""
Streamlit ML Classification Models Comparison App
Allows users to upload test data and see predictions from 5 different classifiers
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, matthews_corrcoef,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 ML Classification Models Comparison")
st.markdown("""
This application demonstrates 5 different machine learning classification models trained on the 
**NPHA Doctor Visits Dataset**. Upload your test data to see predictions and performance metrics.
""")

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page:", ["Home", "Model Comparison", "Make Predictions", "About"])

# Load models and metrics
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    artifact_layouts = {
        BASE_DIR / 'models': {
            'Logistic Regression': 'logistic_regression.joblib',
            'Decision Tree': 'decision_tree.joblib',
            'KNN': 'knn.joblib',
            'Naive Bayes': 'naive_bayes.joblib',
            'SVM': 'svm.joblib',
            'scaler': 'scaler.joblib'
        },
        BASE_DIR / 'model': {
            'Logistic Regression': 'Logistic_Regression.pkl',
            'Decision Tree': 'Decision_Tree.pkl',
            'KNN': 'KNN.pkl',
            'Naive Bayes': 'Naive_Bayes.pkl',
            'SVM': 'SVM.pkl',
            'scaler': 'scaler.pkl'
        }
    }

    for model_dir, files in artifact_layouts.items():
        if all((model_dir / file_name).exists() for file_name in files.values()):
            scaler = joblib.load(model_dir / files['scaler'])
            models = {
                name: joblib.load(model_dir / file_name)
                for name, file_name in files.items()
                if name != 'scaler'
            }
            return models, scaler

    raise FileNotFoundError('No complete trained-model set found in models/ or model/.')

@st.cache_resource
def load_metrics():
    json_path = BASE_DIR / 'models' / 'metrics.json'
    if json_path.exists():
        with json_path.open('r', encoding='utf-8') as metrics_file:
            return json.load(metrics_file)

    csv_path = BASE_DIR / 'model_results.csv'
    if csv_path.exists():
        metrics_df = pd.read_csv(csv_path).set_index('Model')
        metrics_df.columns = metrics_df.columns.str.lower()
        return metrics_df.to_dict(orient='index')

    raise FileNotFoundError('No metrics file found at models/metrics.json or model_results.csv.')

try:
    models, scaler = load_models()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Run the training script first and keep its model artifacts in model/ or models/.")
    st.stop()

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "Home":
    st.header("Welcome to ML Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dataset Information")
        st.write("""
        **Dataset:** NPHA Doctor Visits
        - **Total Samples:** 500+
        - **Features:** 14 health and demographic variables
        - **Target:** Number of Doctors Visited (binary: Low/High)
        - **Data Split:** 80% training, 20% testing
        """)
    
    with col2:
        st.subheader("🎯 Models Implemented")
        models_info = """
        1. **Logistic Regression** - Linear classification
        2. **Decision Tree** - Tree-based classifier
        3. **K-Nearest Neighbor** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier
        5. **Support Vector Machine** - Kernel-based classifier
        """
        st.write(models_info)
    
    st.markdown("---")
    
    st.subheader("📊 Quick Model Metrics Comparison")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(4)
    
    # Create a more visual representation
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax, rot=45)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📈 Metrics Explanation")
    metrics_info = {
        "Accuracy": "Percentage of correct predictions out of all predictions",
        "AUC Score": "Area Under the Receiver Operating Characteristic Curve (0-1, higher is better)",
        "Precision": "Accuracy of positive predictions (True Positives / (TP + FP))",
        "Recall": "Coverage of actual positive cases (True Positives / (TP + FN))",
        "F1 Score": "Harmonic mean of Precision and Recall (0-1, higher is better)",
        "MCC": "Matthews Correlation Coefficient (-1 to 1, higher is better)"
    }
    
    for metric, explanation in metrics_info.items():
        st.write(f"**{metric}:** {explanation}")

# ============================================================================
# PAGE 2: MODEL COMPARISON
# ============================================================================
elif page == "Model Comparison":
    st.header("🔍 Detailed Model Comparison")
    
    # Display metrics table
    st.subheader("Performance Metrics Table")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Download option
    csv = metrics_df.to_csv()
    st.download_button(
        label="📥 Download Metrics as CSV",
        data=csv,
        file_name="model_metrics.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Select metrics to visualize
    st.subheader("📊 Visualization")
    selected_metrics = st.multiselect(
        "Select metrics to visualize:",
        metrics_df.columns.tolist(),
        default=['accuracy', 'f1', 'auc']
    )
    
    if selected_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df[selected_metrics].plot(kind='bar', ax=ax, rot=45)
        ax.set_title('Selected Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Model rankings
    st.markdown("---")
    st.subheader("🏆 Model Rankings by Metric")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Top 3 - Accuracy**")
        top_accuracy = metrics_df['accuracy'].nlargest(3)
        for idx, (model, score) in enumerate(top_accuracy.items(), 1):
            st.write(f"{idx}. {model}: {score:.4f}")
    
    with col2:
        st.write("**Top 3 - F1 Score**")
        top_f1 = metrics_df['f1'].nlargest(3)
        for idx, (model, score) in enumerate(top_f1.items(), 1):
            st.write(f"{idx}. {model}: {score:.4f}")
    
    with col3:
        st.write("**Top 3 - AUC**")
        top_auc = metrics_df['auc'].nlargest(3)
        for idx, (model, score) in enumerate(top_auc.items(), 1):
            st.write(f"{idx}. {model}: {score:.4f}")

# ============================================================================
# PAGE 3: MAKE PREDICTIONS
# ============================================================================
elif page == "Make Predictions":
    st.header("🔮 Make Predictions on Test Data")
    
    # Upload CSV file
    st.subheader("📤 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with test data (must have all 14 features and target column)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df_test = pd.read_csv(uploaded_file)
            st.success(f"✓ File uploaded successfully! Shape: {df_test.shape}")
            
            # Display uploaded data
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df_test.head(), use_container_width=True)
            
            # Check if target column exists
            if 'Number of Doctors Visited' in df_test.columns:
                has_target = True
                y_test = df_test['Number of Doctors Visited'].values
                y_test_binary = (y_test >= 3).astype(int)
                X_test = df_test.drop('Number of Doctors Visited', axis=1)
            else:
                has_target = False
                X_test = df_test
                st.warning("⚠️ Target column 'Number of Doctors Visited' not found. Predictions only (no metrics).")
            
            # Check if all required features are present
            feature_names = scaler.get_feature_names_out().tolist()
            missing_features = [f for f in feature_names if f not in X_test.columns]
            
            if missing_features:
                st.error(f"❌ Missing features: {missing_features}")
                st.stop()
            
            # Prepare data
            X_test_selected = X_test[feature_names]
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Select model
            st.subheader("🤖 Select Model for Prediction")
            selected_model = st.selectbox(
                "Choose a classification model:",
                list(models.keys())
            )
            
            # Make predictions
            if st.button("🚀 Make Predictions", use_container_width=True):
                model = models[selected_model]
                y_pred = model.predict(X_test_scaled)
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = None
                
                # Display predictions
                st.subheader(f"📊 Predictions from {selected_model}")
                
                predictions_df = X_test_selected.copy()
                predictions_df['Predicted'] = y_pred
                if y_pred_proba is not None:
                    predictions_df['Confidence'] = (y_pred_proba * 100).round(2)
                
                st.dataframe(predictions_df, use_container_width=True)
                
                # Download predictions
                csv_pred = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv_pred,
                    file_name=f"predictions_{selected_model.lower()}.csv",
                    mime="text/csv"
                )
                
                # Calculate and display metrics if target is available
                if has_target:
                    st.markdown("---")
                    st.subheader("📈 Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    accuracy = accuracy_score(y_test_binary, y_pred)
                    precision = precision_score(y_test_binary, y_pred)
                    recall = recall_score(y_test_binary, y_pred)
                    f1 = f1_score(y_test_binary, y_pred)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    # Confusion Matrix
                    st.subheader("🔲 Confusion Matrix")
                    cm = confusion_matrix(y_test_binary, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Low', 'High'],
                               yticklabels=['Low', 'High'])
                    ax.set_title(f'Confusion Matrix - {selected_model}', fontweight='bold')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test_binary, y_pred, 
                                                  target_names=['Low', 'High'],
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
                    
                    # AUC and MCC
                    if y_pred_proba is not None:
                        auc = roc_auc_score(y_test_binary, y_pred_proba)
                        st.metric("AUC Score", f"{auc:.4f}")
                    
                    mcc = matthews_corrcoef(y_test_binary, y_pred)
                    st.metric("Matthews Correlation Coefficient (MCC)", f"{mcc:.4f}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================
elif page == "About":
    st.header("ℹ️ About This Application")
    
    st.subheader("📚 Project Overview")
    st.write("""
    The application implements and compares 5 different classification algorithms on the
    NPHA Doctor Visits dataset.
    """)
    
    st.subheader("🎯 Objectives")
    objectives = """
    1. Implement multiple machine learning classification models
    2. Evaluate models using standard performance metrics
    3. Compare model performance across different metrics
    4. Deploy models in an interactive web application
    5. Demonstrate real-world ML deployment workflow
    """
    st.write(objectives)
    
    st.subheader("📊 Dataset Details")
    dataset_info = """
    **Dataset:** NPHA-doctor-visits.csv
    
    **Features (14 total):**
    - Age
    - Physical Health
    - Mental Health
    - Dental Health
    - Employment Status
    - Sleep Factors (6 features):
      - Stress Keeps Patient from Sleeping
      - Medication Keeps Patient from Sleeping
      - Pain Keeps Patient from Sleeping
      - Bathroom Needs Keep Patient from Sleeping
      - Unknown Keeps Patient from Sleeping
      - Trouble Sleeping
    - Prescription Sleep Medication
    - Race
    - Gender
    
    **Target Variable:**
    - Number of Doctors Visited (converted to binary: Low/High)
    """
    st.write(dataset_info)
    
    st.subheader("🧠 Models Implemented")
    models_description = {
        "Logistic Regression": "A linear model for binary classification based on logistic function",
        "Decision Tree": "A tree-based model that creates decision rules from features",
        "K-Nearest Neighbor": "An instance-based learning algorithm using distance metrics",
        "Naive Bayes": "A probabilistic classifier based on Bayes' theorem",
        "Support Vector Machine": "A kernel-based model for linear and non-linear classification"
    }
    
    for model_name, description in models_description.items():
        st.write(f"**{model_name}:** {description}")
    
    st.subheader("📈 Evaluation Metrics")
    st.write("""
    - **Accuracy:** Proportion of correct predictions
    - **AUC Score:** Area under the ROC curve
    - **Precision:** Correctness of positive predictions
    - **Recall:** Coverage of actual positive cases
    - **F1 Score:** Harmonic mean of Precision and Recall
    - **MCC (Matthews Correlation Coefficient):** Balanced measure for binary classification
    """)
    
    st.subheader("👨‍💻 Technical Stack")
    tech_stack = """
    - **Python 3.8+**
    - **Streamlit** - Web application framework
    - **scikit-learn** - Machine learning library
    - **pandas** - Data manipulation
    - **NumPy** - Numerical computing
    - **Matplotlib & Seaborn** - Data visualization
    """
    st.write(tech_stack)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p> Machine Learning Classification Models Comparison </p>
</div>
""", unsafe_allow_html=True)
