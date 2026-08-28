# Online Recruitment Fraud (ORF) Detection System

This repository contains an Online Recruitment Fraud (ORF) detection pipeline designed to scrape live job postings, predict their likelihood of being fraudulent using a trained machine learning model, and provide explainable alerts using SHAP value attribution.

## Project Structure

- `live_scraper.py`: Scrapes job postings from job boards (e.g., Remote.co) or falls back to mock examples.
- `train_model.py`: Preprocesses the job postings dataset, extracts RoBERTa embeddings, uses SMOTE to balance the classes, and trains an MLP classifier.
- `predict_and_alert.py`: Runs the detection pipeline: loads the trained model, scrapes live job listings, classifies them, and runs SHAP explainability analysis on suspicious listings to identify red-flag words.
- `download_emscad.py`: Downloads the EMSCAD dataset (`fake_job_postings.csv`) if it doesn't already exist.
- `generate_dataset.py`: Generates a synthetic balanced dataset of 500 job postings (`job_postings.csv`) for testing.

---

## How to Run

Follow these steps to run the pipeline on Windows.

### 1. Open your terminal
Ensure you are in the project root directory:
```powershell
cd d:\ORF
```

### 2. Activate the virtual environment
The workspace already has a virtual environment (`.venv`) configured with all dependencies. Activate it in PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
*(If using Command Prompt, run `.venv\Scripts\activate.bat` instead.)*

### 3. Run the Inference & Explainability Pipeline (Directly)
Since a pre-trained model is already saved in the `orf_model` directory, you can run the live scraper and prediction system immediately:
```powershell
python predict_and_alert.py
```
This script will:
- Load the pre-trained `orf_model`.
- Scrape or generate job postings.
- Analyze each job posting and print its fraud probability.
- If a high risk of fraud (> 80%) is detected, it will run SHAP explainability to highlight the words contributing most to that prediction (e.g., "Telegram", "WhatsApp", "check") and print a candidate warning email draft.

---

## Retraining the Model (Optional)

If you want to retrain the model on the full dataset:

### 1. (Optional) Re-download the dataset
The full 50MB `fake_job_postings.csv` is already present. If you ever need to download it again:
```powershell
python download_emscad.py
```

### 2. Run the training script
Train the model with a custom sample size to optimize execution time (e.g., 2,000 samples is the default):
```powershell
python train_model.py --sample_size 2000
```
This will:
- Extract text embeddings using RoBERTa.
- Apply SMOTE to handle class imbalance.
- Train the MLP Classifier.
- Save the tokenizer and trained weights to the `orf_model` directory.
