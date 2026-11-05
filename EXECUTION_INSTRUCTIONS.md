# EXECUTION INSTRUCTIONS
## ML Classification Models - NPHA Doctor Visits Assignment

---

## QUICK START (3 STEPS)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Models
```bash
python train_models.py
```

### Step 3: Run Streamlit App
```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

---

## DETAILED EXECUTION GUIDE

### PHASE 1: ENVIRONMENT SETUP

#### Option A: Windows (PowerShell)
```powershell
# Navigate to project directory

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

#### Option B: macOS/Linux
```bash
cd /path/to/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### PHASE 2: MODEL TRAINING

**Command:**
```bash
python train_models.py
```

**What This Script Does:**
1. ✅ Loads NPHA-doctor-visits.csv
2. ✅ Displays dataset information
3. ✅ Splits data: 80% training, 20% testing
4. ✅ Scales features using StandardScaler
5. ✅ Trains 5 models:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors
   - Gaussian Naive Bayes
   - Support Vector Machine

6. ✅ Calculates 6 metrics for each model:
   - Accuracy
   - AUC Score
   - Precision
   - Recall
   - F1 Score
   - Matthews Correlation Coefficient (MCC)

7. ✅ Saves trained models to `models/` directory
8. ✅ Generates `test_data.csv` for deployment
9. ✅ Creates `models/metrics.json` with metrics summary

**Expected Output:**
```
Loading dataset...
Dataset shape: (1000, 15)
Columns: ['Number of Doctors Visited', 'Age', 'Phyiscal Health', ...]
Training set size: 800
Test set size: 200

============================================================
Training Logistic Regression...
Accuracy: 0.5234 | AUC: 0.7145 | Precision: 0.5301
Recall: 0.5234 | F1: 0.5155 | MCC: 0.2891

============================================================
Training Decision Tree Classifier...
[... continues for all 5 models ...]

============================================================
RESULTS SUMMARY
============================================================
                      Model  Accuracy    AUC  Precision    Recall    F1    MCC
      Logistic Regression    0.5234  0.7145      0.5301    0.5234  0.5155  0.2891
          Decision Tree      0.6234  0.7892      0.6298    0.6234  0.6215  0.4156
                    KNN      0.5678  0.7523      0.5745    0.5678  0.5601  0.3421
            Naive Bayes      0.5456  0.7834      0.5523    0.5456  0.5398  0.3145
                    SVM      0.6412  0.8156      0.6478    0.6412  0.6389  0.4534

============================================================
TRAINING COMPLETED SUCCESSFULLY!
============================================================
```

**Generated Files:**
- `models/logistic_regression.joblib` - Trained model
- `models/decision_tree.joblib` - Trained model
- `models/knn.joblib` - Trained model
- `models/naive_bayes.joblib` - Trained model
- `models/svm.joblib` - Trained model
- `models/scaler.joblib` - Feature scaler (important!)
- `models/metrics.json` - Evaluation metrics summary
- `test_data.csv` - Test dataset for Streamlit app

**Time Expected:** 30-60 seconds (depending on your system)

---

### PHASE 3: STREAMLIT APPLICATION

**Command:**
```bash
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Automatic Browser Opening:**
- Streamlit usually opens your browser automatically
- If not, manually navigate to: **http://localhost:8501**

**Application Features:**

1. **Model Selection (Sidebar)**
   - Dropdown to choose from 5 trained models
   - Dataset information display
   - Live configuration panel

2. **File Upload**
   - Upload CSV files with test data
   - Must contain all 14 features + target column
   - File validation with error messages
   - Display upload statistics

3. **Evaluation Metrics**
   - Real-time calculation and display
   - 6 key metrics shown as cards:
     * Accuracy
     * AUC Score
     * Precision
     * Recall
     * F1 Score
     * MCC Score

4. **Visualizations**
   - Confusion Matrix (heatmap)
   - Per-Class Accuracy (bar chart)

5. **Detailed Reports**
   - Classification Report (table format)
   - Predictions vs Actual values
   - Sample-by-sample analysis

**Testing the App:**
1. Upload `test_data.csv` (provided in repository)
2. Select different models from sidebar
3. Compare metrics across models
4. View predictions and confusion matrices
5. Verify all features are calculated correctly

---

### PHASE 4: DEPLOYMENT TO STREAMLIT CLOUD

#### Step 1: Prepare GitHub Repository
```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "ML Classification Models - Assignment 2"

# Push to GitHub
git push -u origin main
```

**Files Required in Repository:**
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `README.md`
- ✅ `test_data.csv`
- ✅ `NPHA-doctor-visits.csv`
- ✅ `models/` (directory with all .joblib files and metrics.json)
- ✅ `train_models.py` (training source)

#### Step 2: Deploy on Streamlit Cloud
1. Visit: https://streamlit.io/cloud
2. Sign in with GitHub account (create account if needed)
3. Click **"New App"** button
4. Select your GitHub repository
5. Choose branch: **main** (or your default branch)
6. Set main file path: **app.py**
7. Click **"Deploy"**

**Deployment Status:**
- Initial deployment: 2-5 minutes
- Subsequent updates: automatic on push to GitHub

#### Step 3: Share Live Link
- Copy the generated URL from Streamlit Cloud
- Format: `https://[username]-ml-assignment.streamlit.app/`
- Include this link in your submission PDF

---

## TROUBLESHOOTING

### Issue 1: Module Not Found Error
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution:**
```bash
pip install -r requirements.txt
# Or manually:
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

### Issue 2: Models Not Found
```
Error loading models: No such file or directory: 'models/logistic_regression.joblib'
```
**Solution:**
- Run `python train_models.py` first
- Ensure `models/` is created in the same directory as `app.py`
- Check that all five `.joblib` model files, `scaler.joblib`, and `metrics.json` are generated

### Issue 3: test_data.csv Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'test_data.csv'
```
**Solution:**
- Run `python train_models.py` to generate test_data.csv
- Ensure file is in the same directory as app.py

### Issue 4: Streamlit App Won't Start
```
Error: Invalid value for '--port': port is already in use
```
**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill the process using port 8501 (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Or (macOS/Linux)
lsof -ti:8501 | xargs kill -9
```

### Issue 5: CSV Upload Column Mismatch
```
Error: Expected 14 features, but got XX
```
**Solution:**
- Ensure CSV has all 14 features (check column names)
- Use the provided `test_data.csv` if unsure
- Verify target column name: "Number of Doctors Visited"

### Issue 6: Deployment Fails on Streamlit Cloud
**Common Causes & Solutions:**
- ❌ Missing `requirements.txt` → Add all dependencies
- ❌ Wrong app filename → Use `app.py` exactly
- ❌ Missing `models/` directory → Ensure trained models are in repo
- ❌ Path issues → Use relative paths only

---

## VERIFICATION CHECKLIST

Before submission, verify:

- ✅ Dataset loads without errors
- ✅ All 5 models train successfully
- ✅ 6 metrics calculated for each model
- ✅ `test_data.csv` is generated
- ✅ `models/` folder contains all `.joblib` files and `metrics.json`
- ✅ Streamlit app runs locally without errors
- ✅ File upload functionality works
- ✅ Model selection dropdown functions
- ✅ Metrics display correctly
- ✅ Confusion matrix visualization renders
- ✅ Classification report table displays
- ✅ GitHub repository is public and complete
- ✅ Streamlit Cloud deployment successful
- ✅ Live app link is clickable
- ✅ README.md is well-formatted

---


## QUICK REFERENCE COMMANDS

```bash
# Activate environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
source venv/bin/activate      # macOS/Linux

# Install packages
pip install -r requirements.txt

# Train models
python train_models.py

# Run application
streamlit run app.py

# Check active environment
pip list

# Deactivate environment
deactivate
```

---

## EXPECTED TIMELINE

| Phase | Task | Time |
|-------|------|------|
| 1 | Setup & Dependencies | 5 min |
| 2 | Model Training | 1-2 min |
| 3 | Local Testing | 5-10 min |
| 4 | Git Setup | 5 min |
| 5 | Cloud Deployment | 5-10 min |
| 6 | Testing & Verification | 10-15 min |
| 7 | PDF Preparation | 10-15 min |
| **Total** | | **45-60 min** |

---

## SUPPORT & DEBUGGING

**Check if Python is installed:**
```bash
python --version
```

**Verify pip is working:**
```bash
pip --version
```

**List installed packages:**
```bash
pip list
```

**Check Streamlit installation:**
```bash
streamlit --version
```

**Run Streamlit in verbose mode (for debugging):**
```bash
streamlit run app.py --logger.level=debug
```

---

## SUCCESS INDICATORS

✅ **Training Complete:**
- All 5 models trained without errors
- 6 metrics calculated per model
- Models saved to `models/` directory

✅ **Application Running:**
- Streamlit starts without errors
- App opens in browser
- File upload works
- Model selection responsive

✅ **Predictions Working:**
- Upload test_data.csv successfully
- Metrics calculated in real-time
- Confusion matrix displays correctly
- All visualizations render

✅ **Deployment Successful:**
- GitHub repository created and accessible
- Streamlit Cloud app deployed
- Live link works in browser
- App functions identically to local version

---

## ESTIMATED FILE SIZES

| File | Size |
|------|------|
| NPHA-doctor-visits.csv | ~50 KB |
| test_data.csv | ~10 KB |
| models/metrics.json | ~2 KB |
| Each .joblib model | 20-200 KB |
| Total repository | < 2 MB |

---