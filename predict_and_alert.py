# type: ignore
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import shap
from live_scraper import LiveScraper
import warnings

# Global variables for models
tokenizer = None
roberta_model = None
clf = None
device = None

def predict_probs(texts):
    """Prediction pipeline mapping raw texts to class probabilities.
    
    SHAP will use this function to perturb text features and calculate attributions.
    """
    global tokenizer, roberta_model, clf, device
    
    # SHAP might pass numpy array of strings
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
        
    # Ensure all elements are clean strings
    texts = [str(t) for t in texts]
    
    # Tokenize input texts
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        # Extract CLS token representations
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    # Get probabilities from trained classifier head
    probs = clf.predict_proba(embeddings)
    return probs

def notify_candidate(title, company, fraud_score, top_keywords, url):
    """Format and display candidate security warning email draft."""
    keywords_str = ', '.join([f"'{kw}'" for kw in top_keywords if kw])
    email_draft = f"""
======================================================================
*** SECURITY ALERT SYSTEM TRIGGERED ***
======================================================================
High Fraud Probability Detected: {fraud_score * 100:.2f}%
Job Title: {title}
Company: {company}
Source URL: {url}

[DRAFT WARNING EMAIL TO APPLICANT]
Subject: URGENT: Security Alert Regarding Job Application for '{title}'

Dear Candidate,

Our automated Online Recruitment Fraud (ORF) detection system has flagged the job listing "{title}" at "{company}" as highly suspicious.

We care about your safety. Our machine learning system analyzed the job description and highlighted the following text elements as high-risk scam indicators:
>>> {keywords_str} <<<

Please be extremely cautious:
1. Legitimate employers will rarely conduct official interviews via personal chat apps like Telegram or WhatsApp.
2. Never pay upfront fees for processing, software licensing, or background checks.
3. Be suspicious of requests to deposit checks and immediately wire money to "approved equipment vendors."
4. If you have already shared banking information, please contact your financial institution immediately.

Stay safe,
Recruitment Security Team
======================================================================
"""
    print(email_draft)

def main():
    global tokenizer, roberta_model, clf, device
    
    model_path = "./orf_model"
    if not os.path.exists(model_path):
        model_path = "d:/ORF/orf_model"
        
    if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "classifier.joblib")):
        print(f"Model directory or classifier not found. Please run train_model.py first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models onto {device}...")
    
    # Load tokenizers and configuration from saved path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model from original weights (uses config saved in path)
    model_name = "distilroberta-base"
    roberta_model = AutoModel.from_pretrained(model_name).to(device)
    
    # Load classifier head
    clf = joblib.load(os.path.join(model_path, "classifier.joblib"))
    print("Models loaded successfully.")
    
    # 1. Initialize the SHAP Explainer
    # We use partition-based Explainer, passing our prediction function and the text masker
    print("\nInitializing SHAP Explainer (this may take a few seconds)...")
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_probs, masker=masker)
    print("SHAP Explainer ready.")

    # 2. Run Live Scraper
    scraper = LiveScraper()
    jobs = scraper.scrape_jobs(max_jobs=4)
    
    print(f"\nAnalyzing {len(jobs)} job postings for fraud...")
    
    for idx, job in enumerate(jobs):
        title = job.get('title', 'Untitled')
        company = job.get('company_profile', 'Unknown Company').split('.')[0] # Get first sentence or name
        if not company or len(company) > 100:
            company = "Company Details"
            
        print(f"\n[{idx+1}/{len(jobs)}] Analyzing: '{title}'...")
        
        # Combine fields in the exact order as training
        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        combined_text = " ".join([job.get(col, '') for col in text_cols])
        combined_text = ' '.join(combined_text.split())
        
        # Get probability
        probs = predict_probs([combined_text])[0]
        # Class 1 is Fraudulent
        fraud_score = probs[1]
        
        print(f"  -> Fraud Probability: {fraud_score * 100:.2f}%")
        
        if fraud_score > 0.8:
            print("  [ALERT] High risk of fraud! Running SHAP word attributions...")
            
            # Compute SHAP values for this text
            shap_values = explainer([combined_text])
            
            # Extract contributions for class 1 (Fraudulent)
            contributions = shap_values.values[0, :, 1]
            feature_names = shap_values.data[0]
            
            # Get top indices contributing positively
            top_indices = np.argsort(contributions)[-6:][::-1]
            
            # Process and clean tokens (handling RoBERTa's 'Ġ' marker)
            seen_keywords = set()
            top_keywords = []
            keyword_contributions = []
            
            for i in top_indices:
                val = contributions[i]
                if val <= 0:
                    continue
                word = feature_names[i].replace('Ġ', '').replace('</w>', '').strip()
                # Exclude punctuation, short tokens, or duplicates
                if word and word.isalnum() and len(word) > 2 and word.lower() not in seen_keywords:
                    seen_keywords.add(word.lower())
                    top_keywords.append(word)
                    keyword_contributions.append(val)
                    
            # Print contributing factors in console
            print("\n  [SHAP EXPLANATION HIGHLIGHTS] Top Scam Indicators:")
            for word, val in zip(top_keywords[:3], keyword_contributions[:3]):
                print(f"    - \"{word}\" (SHAP contribution: +{val:.3f})")
                
            # Trigger warning notification
            notify_candidate(
                title=title, 
                company=company, 
                fraud_score=fraud_score, 
                top_keywords=top_keywords[:3],
                url=job.get('url', 'N/A')
            )
        else:
            print("  [OK] Job listing appears legitimate. No actions required.")

if __name__ == "__main__":
    # Ignore deprecation or user warnings from packages (e.g., HuggingFace/SHAP)
    warnings.filterwarnings("ignore")
    main()
