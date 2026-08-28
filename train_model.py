# type: ignore
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, auc
import joblib
import os
from tqdm import tqdm
import argparse

def extract_embeddings(texts, model, tokenizer, batch_size=32, device='cpu'):
    """Extract CLS token embeddings from pre-trained RoBERTa for a list of texts."""
    embeddings = []
    model.eval()
    
    # Process texts in batches to manage memory
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch = list(texts[i:i+batch_size])
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Extracted embedding for CLS token (index 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
            
    return np.array(embeddings)

def main():
    parser = argparse.ArgumentParser(description="Train ORF model using RoBERTa, SMOTE, and MLP")
    parser.add_argument('--sample_size', type=int, default=2000, 
                        help='Number of samples to use for fast execution (use high number or -1 for all)')
    args = parser.parse_args()

    data_path = "fake_job_postings.csv"
    if not os.path.exists(data_path):
        data_path = "d:/ORF/fake_job_postings.csv"
        
    if not os.path.exists(data_path):
        print(f"Dataset not found. Please run download_emscad.py first.")
        return
        
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Preprocessing
    # Combine text columns representing the job posting
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            df[col] = ''
            
    df['text'] = df[text_cols].astype(str).agg(' '.join, axis=1)
    
    # Clean text to remove excess whitespace
    df['text'] = df['text'].apply(lambda x: ' '.join(x.split()))
    
    # Filter out records where text is extremely short or empty
    df = df[df['text'].str.strip() != '']
    
    # Stratified Sampling for local performance
    if args.sample_size > 0 and len(df) > args.sample_size:
        print(f"Sampling dataset to {args.sample_size} records to optimize training time...")
        # Maintain class distribution (approx 95% legit, 5% fraud)
        legit_df = df[df['fraudulent'] == 0]
        fraud_df = df[df['fraudulent'] == 1]
        
        legit_ratio = len(legit_df) / len(df)
        fraud_ratio = len(fraud_df) / len(df)
        
        n_legit = int(args.sample_size * legit_ratio)
        n_fraud = int(args.sample_size * fraud_ratio)
        
        # Ensure we have at least some samples from both classes
        n_legit = max(1, n_legit)
        n_fraud = max(1, n_fraud)
        
        legit_sampled = legit_df.sample(n=n_legit, random_state=42)
        fraud_sampled = fraud_df.sample(n=n_fraud, random_state=42)
        
        df_sampled = pd.concat([legit_sampled, fraud_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df_sampled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    print(f"Dataset size for training: {len(df_sampled)} (Legitimate: {sum(df_sampled['fraudulent']==0)}, Fraudulent: {sum(df_sampled['fraudulent']==1)})")

    # 3. Train-Test Split (Stratified to maintain class ratios)
    print("Splitting into train and test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_sampled['text'].tolist(),
        df_sampled['fraudulent'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_sampled['fraudulent'].tolist()
    )
    
    # 4. RoBERTa Embedding Extraction
    model_name = "distilroberta-base"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for embedding extraction.")
    
    print(f"Loading {model_name} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    roberta_model = AutoModel.from_pretrained(model_name).to(device)
    
    print("Extracting train embeddings...")
    X_train_embeddings = extract_embeddings(X_train_text, roberta_model, tokenizer, device=device)
    
    print("Extracting test embeddings...")
    X_test_embeddings = extract_embeddings(X_test_text, roberta_model, tokenizer, device=device)
    
    # 5. Apply SMOTE to training embeddings to balance classes
    print("\nApplying SMOTE to balance the training set embeddings...")
    print(f"Original class counts: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_embeddings, y_train)
    print(f"SMOTE balanced class counts: {np.bincount(y_train_resampled)}")
    
    # 6. Train MLP Classifier Head on Balanced Embeddings
    print("\nTraining Multi-Layer Perceptron (MLP) classification head...")
    # hidden_layer_sizes=(128, 64) provides non-linear classification boundary
    # early_stopping=True prevents overfitting on the synthetic/interpolated representations
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=200, 
        random_state=42, 
        early_stopping=True,
        validation_fraction=0.1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    print("Training finished!")
    
    # 7. Evaluate on the Original (Unbalanced) Test Set
    print("\nEvaluating model on the original test set (stratified and imbalanced)...")
    y_pred = clf.predict(X_test_embeddings)
    y_prob = clf.predict_proba(X_test_embeddings)[:, 1]
    
    # Calculate Precision-Recall Area Under Curve (PR-AUC)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print("                MODEL METRICS REPORT")
    print("="*50)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"PR-AUC (Precision-Recall Area Under Curve): {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))
    print("="*50)
    
    # 8. Save Models and Configs
    save_dir = "./orf_model"
    if not os.path.exists(save_dir) and os.path.exists("d:/ORF"):
        save_dir = "d:/ORF/orf_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the tokenizer and base model config for inference pipeline
    tokenizer.save_pretrained(save_dir)
    roberta_model.config.save_pretrained(save_dir)
    
    # Save the MLP classifier head
    clf_path = os.path.join(save_dir, "classifier.joblib")
    joblib.dump(clf, clf_path)
    
    print(f"\nModel and configurations successfully saved to {save_dir}")

if __name__ == "__main__":
    main()
