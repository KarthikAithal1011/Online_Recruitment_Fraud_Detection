# type: ignore
import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
from live_scraper import LiveScraper
import uvicorn
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Online Recruitment Fraud (ORF) Detection API")

# Global variables for model assets
tokenizer = None
roberta_model = None
clf = None
explainer = None
device = None

# Model directory configuration
model_path = "./orf_model"
if not os.path.exists(model_path) and os.path.exists("d:/ORF/orf_model"):
    model_path = "d:/ORF/orf_model"

def load_model_assets():
    global tokenizer, roberta_model, clf, explainer, device
    
    if clf is not None:
        return # Assets already loaded
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models onto {device}...")
    
    if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "classifier.joblib")):
        raise RuntimeError("Trained model assets not found in orf_model directory. Please train the model first.")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_name = "distilroberta-base"
    roberta_model = AutoModel.from_pretrained(model_name).to(device)
    clf = joblib.load(os.path.join(model_path, "classifier.joblib"))
    
    # Initialize SHAP explainer
    print("Initializing SHAP Explainer...")
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_probs_internal, masker=masker)
    print("Model assets and SHAP Explainer loaded successfully.")

def predict_probs_internal(texts):
    """Internal prediction function for classification and SHAP perturbations."""
    global tokenizer, roberta_model, clf, device
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    texts = [str(t) for t in texts]
    
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    probs = clf.predict_proba(embeddings)
    return probs

# Request schemas
class JobAnalysisRequest(BaseModel):
    title: str
    company_profile: Optional[str] = ""
    description: str
    requirements: Optional[str] = ""
    benefits: Optional[str] = ""
    url: Optional[str] = ""

class CustomTextRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    try:
        load_model_assets()
    except Exception as e:
        print(f"Error loading model assets at startup: {e}")
        # We don't crash startup so user can see backend page, 
        # but API routes will verify load status.

@app.get("/api/status")
def get_status():
    global clf
    return {
        "model_loaded": clf is not None,
        "device": str(device) if device else "not loaded",
        "model_path": model_path
    }

@app.get("/api/scrape")
def scrape_jobs(limit: int = 5):
    """Scrape job postings from Remote.co or fallback to mock postings."""
    try:
        scraper = LiveScraper()
        jobs = scraper.scrape_jobs(max_jobs=limit)
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

def is_valid_job_text(text: str) -> tuple[bool, str]:
    """Validate if the text appears to be a legitimate job description.
    Checks for programming code patterns, word count, character count, and job-related keywords.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return False, "Input text cannot be empty."
        
    # Detect Markdown code block syntax
    if "```" in text:
        return False, "Input contains code blocks (```). Code files or scripts are not allowed."
        
    # Detect common programming keywords & syntax patterns
    code_patterns = [
        r'\bdef\s+\w+\s*\(',                      # def func(
        r'\bclass\s+\w+\s*[:{]',                   # class Foo: or class Foo {
        r'\bimport\s+[\w\s,]+',                    # import module
        r'\bfrom\s+\w+\s+import\b',                # from module import ...
        r'\bconst\s+\w+\s*=',                      # const x =
        r'\blet\s+\w+\s*=',                        # let x =
        r'\bvar\s+\w+\s*=',                        # var x =
        r'\bfunction\s+\w*\s*\(',                  # function() or function foo()
        r'#include\s*[<"]',                        # #include <stdio.h>
        r'\bpublic\s+(static\s+)?(class|void|int|double|String)\b', # Java/C# signatures
        r'console\.log\s*\(',                      # console.log(
        r'print\s*\([^)]*\)',                      # print("foo") or print(x)
        r'<\?php',                                 # php
        r'<\/?[a-z1-6]+(\s+[^>]+)*>',              # HTML tags like <div>, <p>
        r'\bselect\s+.*\s+from\b',                 # SQL select
        r'[{][\s\S]*[}]',                          # Curly brace structures (JSON or JS object)
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False, "Input appears to contain programming code or developer markup. Please paste a standard job description."
            
    # Check for dense code characters
    code_chars = sum(text.count(char) for char in ['{', '}', '[', ']', ';', '<', '>', '=', '(', ')', '_', '$'])
    if len(text) > 0 and (code_chars / len(text)) > 0.08:
        return False, "Input contains a high density of programming symbols. Please input a regular text job description."

    words = cleaned.split()
    if len(words) < 15:
        return False, f"Input is too short (found {len(words)} words, minimum 15 required)."
    if len(cleaned) < 80:
        return False, f"Input is too short (found {len(cleaned)} characters, minimum 80 required)."
        
    # List of job-related keywords to check for context
    job_keywords = {
        'job', 'position', 'role', 'work', 'experience', 'requirement', 'requirements', 
        'responsibility', 'responsibilities', 'duty', 'duties', 'team', 'skill', 'skills', 
        'candidate', 'apply', 'company', 'benefit', 'benefits', 'salary', 'qualification',
        'qualifications', 'employment', 'hire', 'hiring', 'recruitment', 'recruiting'
    }
    
    # Check if at least 2 distinct job-related keywords appear in the lowercased text
    lower_text = cleaned.lower()
    sub_matches = 0
    for kw in job_keywords:
        if kw in lower_text:
            sub_matches += 1
            
    if sub_matches < 2:
        return False, (
            "The input does not appear to be a job posting. Please include job-related "
            "details such as role responsibilities, requirements, skills, or company info."
        )
        
    return True, ""

@app.post("/api/analyze")
def analyze_job(req: JobAnalysisRequest):
    """Analyze a single job posting for recruitment fraud."""
    global clf, explainer
    if clf is None:
        try:
            load_model_assets()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")
            
    # Combine fields in training sequence: title, company_profile, description, requirements, benefits
    combined_text = " ".join([
        req.title,
        req.company_profile or "",
        req.description,
        req.requirements or "",
        req.benefits or ""
    ])
    combined_text = " ".join(combined_text.split()) # Clean spacing
    
    # Validate job text content to ensure it is a job-related text
    is_valid, validation_msg = is_valid_job_text(combined_text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=validation_msg)
        
    try:
        # Get predictions
        probs = predict_probs_internal([combined_text])[0]
        fraud_score = float(probs[1])
        
        # Calculate risk levels
        if fraud_score <= 0.3:
            risk_level = "low"
            is_fraudulent = False
        elif fraud_score <= 0.7:
            risk_level = "medium"
            is_fraudulent = False
        else:
            risk_level = "high"
            is_fraudulent = True
            
        shap_explanations = []
        email_draft = None
        
        # Run SHAP if it's suspicious or fraudulent (> 30%) to show indicators
        if fraud_score > 0.3:
            shap_values = explainer([combined_text])
            contributions = shap_values.values[0, :, 1]
            feature_names = shap_values.data[0]
            
            # Sort indicators by contribution
            top_indices = np.argsort(contributions)[::-1]
            seen_words = set()
            
            for i in top_indices:
                val = float(contributions[i])
                if val <= 0.01: # Only positive contributors
                    continue
                word = feature_names[i].replace('Ġ', '').replace('</w>', '').strip()
                if word and word.isalnum() and len(word) > 2 and word.lower() not in seen_words:
                    seen_words.add(word.lower())
                    shap_explanations.append({
                        "word": word,
                        "score": round(val, 4)
                    })
                    if len(shap_explanations) >= 5: # Limit top 5
                        break
                        
        if risk_level == "high":
            # Generate email draft
            top_keywords = [item["word"] for item in shap_explanations[:3]]
            keywords_str = ', '.join([f"'{kw}'" for kw in top_keywords])
            
            clean_company = (req.company_profile or "Unknown Company").split('.')[0]
            if not clean_company or len(clean_company) > 80:
                clean_company = "Recruitment Poster"
                
            email_draft = f"""Subject: URGENT: Security Alert Regarding Job Application for '{req.title}'
 
Dear Candidate,
 
Our automated Online Recruitment Fraud (ORF) detection system has flagged the job listing "{req.title}" at "{clean_company}" as highly suspicious (Confidence: {fraud_score * 100:.2f}%).
 
Our machine learning model highlighted the following text elements as high-risk scam indicators:
>>> {keywords_str} <<<
 
Please be extremely cautious:
1. Legitimate employers will rarely conduct official interviews via personal chat apps like Telegram or WhatsApp.
2. Never pay upfront fees for processing, software licensing, or background checks.
3. Be suspicious of requests to deposit checks and immediately wire money to "approved equipment vendors."
4. If you have already shared banking information, please contact your financial institution immediately.
 
Stay safe,
Recruitment Security Team"""
 
        return {
            "fraud_probability": round(fraud_score, 4),
            "is_fraudulent": is_fraudulent,
            "risk_level": risk_level,
            "shap_indicators": shap_explanations,
            "email_draft": email_draft
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/api/analyze-custom")
def analyze_custom_text(req: CustomTextRequest):
    """Analyze custom text directly inputted by user."""
    # We map custom text to description, empty other fields
    analysis_req = JobAnalysisRequest(
        title="Custom Text Analysis",
        company_profile="User Input Profile",
        description=req.text,
        requirements="",
        benefits="",
        url="User Input"
    )
    return analyze_job(analysis_req)

@app.get("/mock-job/{job_id}", response_class=HTMLResponse)
def get_mock_job(job_id: int):
    """Retrieve details for a mock job listing and render them in a clean, professional template."""
    scraper = LiveScraper()
    mock_jobs = scraper.get_mock_jobs()
    
    # Try to find the mock job with matching ID
    selected_job = None
    for j in mock_jobs:
        if j.get('job_id') == str(job_id):
            selected_job = j
            break
            
    if not selected_job:
        raise HTTPException(status_code=404, detail="Mock Job Posting Not Found")
        
    # Render a premium recruitment posting page
    company_name = selected_job['company_profile'].split('.')[0] if selected_job.get('company_profile') else "Mock Company Solutions"
    if len(company_name) > 60:
        company_name = "Mock Company Solutions"
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{selected_job['title']} | Mock Job Board</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --bg-main: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.75);
            --border-color: rgba(148, 163, 184, 0.12);
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --accent-cyan: #38bdf8;
            --accent-purple: #6366f1;
            --gradient-purple: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            --gradient-cyan: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
            --font-heading: 'Outfit', 'Plus Jakarta Sans', sans-serif;
            --font-body: 'Plus Jakarta Sans', sans-serif;
            --shadow-main: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: var(--font-body);
        }}
        
        body {{
            background-color: var(--bg-main);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 60px 20px;
            background-image: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.12) 0%, transparent 50%);
        }}
        
        .container {{
            max-width: 850px;
            width: 100%;
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            box-shadow: var(--shadow-main);
            padding: 40px;
        }}
        
        .header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 30px;
            margin-bottom: 30px;
        }}
        
        .badge-portal {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.25);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-radius: 20px;
            margin-bottom: 20px;
        }}
        
        h1 {{
            font-family: var(--font-heading);
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 14px;
            background: linear-gradient(135deg, #fff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .company-info {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 1.15rem;
            color: var(--accent-cyan);
            font-weight: 600;
            margin-bottom: 16px;
        }}
        
        .company-info i {{
            font-size: 1.1rem;
        }}
        
        .meta-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 20px;
        }}
        
        .meta-tag {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 6px 14px;
            border-radius: 12px;
            font-weight: 500;
        }}
        
        .meta-tag i {{
            color: var(--text-muted);
        }}
        
        .section {{
            margin-bottom: 35px;
        }}
        
        .section-title {{
            font-family: var(--font-heading);
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-title i {{
            color: var(--accent-purple);
            font-size: 1.1rem;
        }}
        
        .section-content {{
            font-size: 14px;
            line-height: 1.7;
            color: var(--text-secondary);
            white-space: pre-line;
            text-align: justify;
        }}
        
        .apply-container {{
            border-top: 1px solid var(--border-color);
            padding-top: 30px;
            margin-top: 10px;
            text-align: center;
        }}
        
        .btn-apply {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            background: var(--gradient-purple);
            color: var(--text-primary);
            text-decoration: none;
            padding: 14px 32px;
            font-weight: 700;
            font-size: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(138, 43, 226, 0.3);
            transition: all 0.25s ease;
            cursor: pointer;
            border: none;
            width: 100%;
        }}
        
        .btn-apply:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(138, 43, 226, 0.5);
        }}
        
        .btn-apply:active {{
            transform: translateY(0);
        }}
        
        .info-notice {{
            margin-top: 20px;
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge-portal">
                <i class="fa-solid fa-graduation-cap"></i> Mock Job Portal
            </span>
            <h1>{selected_job['title']}</h1>
            <div class="company-info">
                <i class="fa-regular fa-building"></i>
                <span>{company_name}</span>
            </div>
            <div class="meta-tags">
                <div class="meta-tag">
                    <i class="fa-solid fa-location-dot"></i>
                    <span>Remote / WFH</span>
                </div>
                <div class="meta-tag">
                    <i class="fa-solid fa-clock"></i>
                    <span>Full-Time Position</span>
                </div>
                <div class="meta-tag">
                    <i class="fa-solid fa-globe"></i>
                    <span>Global Applicants Welcome</span>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">
                <i class="fa-solid fa-circle-info"></i> Company Profile
            </div>
            <div class="section-content">
                {selected_job['company_profile']}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">
                <i class="fa-solid fa-align-left"></i> Job Description
            </div>
            <div class="section-content">
                {selected_job['description']}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">
                <i class="fa-solid fa-clipboard-list"></i> Requirements
            </div>
            <div class="section-content">
                {selected_job['requirements']}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">
                <i class="fa-solid fa-gift"></i> Benefits
            </div>
            <div class="section-content">
                {selected_job['benefits']}
            </div>
        </div>
        
        <div class="apply-container">
            <button class="btn-apply" onclick="alert('This is a simulated job posting for security training and demonstration purposes.')">
                <i class="fa-solid fa-paper-plane"></i> Apply for this position
            </button>
            <div class="info-notice">
                <i class="fa-solid fa-shield-halved"></i>
                <span>This page is served locally by the Online Recruitment Fraud Detection Center.</span>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html_content

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
