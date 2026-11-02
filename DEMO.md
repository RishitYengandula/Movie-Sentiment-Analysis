# 🎬 1-2 Minute Project Demo Guide

This guide provides a structured, concise walk-through for demonstrating the **Movie Review Sentiment Analysis** project to an interviewer, recruiter, or internship evaluator.

---

### Step 1: Project Overview (15 seconds)
- Open the [README.md](README.md).
- Highlight key points:
  - Binary NLP classifier trained on the **IMDb 50K** movie review dataset.
  - End-to-end `sklearn.pipeline.Pipeline` with **TF-IDF (unigrams + bigrams)** and **Logistic Regression**.
  - **Zero data leakage**: feature extraction fitted strictly on training data.
  - Test set accuracy: **90.69%**, F1-Score: **90.86%**.

---

### Step 2: Launch the Web Application (10 seconds)
Launch the Streamlit interface from PowerShell:
```powershell
.venv\Scripts\activate
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501`. Note that the app loads the pre-trained artifact (`models/sentiment_pipeline.joblib`) with zero cold-start delay and **never retrains the model in the UI**.

---

### Step 3: Positive Review Demonstration (20 seconds)
1. Click the **`👍 Positive Example`** button.
   - Text: `"I absolutely loved this movie. The acting was fantastic and the story was engaging."`
2. Click **`🔍 Analyze Sentiment`**.
3. Point out:
   - Green card rendering: **Sentiment: Positive**.
   - Model Confidence: **90.88%** with corresponding progress bar.

---

### Step 4: Negative Review Demonstration (20 seconds)
1. Click the **`👎 Negative Example`** button.
   - Text: `"This was a terrible movie. The plot was boring and the acting was awful."`
2. Click **`🔍 Analyze Sentiment`**.
3. Point out:
   - Red card rendering: **Sentiment: Negative**.
   - Model Confidence: **99.91%** with corresponding progress bar.

---

### Step 5: Mixed Review Demonstration (15 seconds)
1. Click the **`⚖️ Mixed / Neutral Example`** button.
   - Text: `"The movie was okay, but nothing particularly special."`
2. Click **`🔍 Analyze Sentiment`**.
3. Note that the model classifies subtle critiques based on learned vocabulary weights (**Sentiment: Negative**, **94.72%** confidence).

---

### Step 6: Architecture & Metrics Sidebar (20 seconds)
- Expand/scroll the left **Sidebar**:
  - Show the model configuration (50k features, sublinear TF, L-BFGS solver).
  - Point out the **Phase 4 Evaluation Metrics table** (90.55% CV accuracy, 90.69% test accuracy).
  - Highlight the embedded **Confusion Matrix** (`reports/confusion_matrix.png`) showing balanced True Positives (4,589) and True Negatives (4,405).

---

### Step 7: Automated Tests & Clean Codebase (20 seconds)
Open the terminal and run:
```powershell
pytest -v
```
Show that **10 out of 10 automated unit tests pass in < 2 seconds**, verifying text normalization, model loading, prediction schemas, and input validation.
