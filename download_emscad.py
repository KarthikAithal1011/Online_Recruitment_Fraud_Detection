# type: ignore
import os
import requests
import pandas as pd

def download_file(url, local_filename):
    print(f"Downloading {url} to {local_filename}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # We use stream=True to handle large files and show progress
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1MB chunks
        
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='\r')
                    else:
                        print(f"Downloaded: {downloaded / (1024*1024):.1f} MB", end='\r')
    print("\nDownload complete!")

def main():
    url = "https://raw.githubusercontent.com/stimils2/Fake-Job-Prediction-Using-BERT/master/fake_job_postings.csv"
    output_path = "d:/ORF/fake_job_postings.csv"
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}.")
    else:
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            # Try a backup URL
            backup_url = "https://raw.githubusercontent.com/RodAli/fake-job-postings-analysis/main/fake_job_postings.csv"
            print("Trying backup URL...")
            try:
                download_file(backup_url, output_path)
            except Exception as backup_e:
                print(f"Backup download failed: {backup_e}")
                return

    # Load and print statistics
    print("Loading dataset to analyze...")
    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("\n" + "="*40)
    print("       EMSCAD DATASET STATISTICS")
    print("="*40)
    print(f"Total Rows (Job Postings): {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print("\nColumns list:")
    print(list(df.columns))
    
    # Class balance
    if 'fraudulent' in df.columns:
        counts = df['fraudulent'].value_counts()
        legit_count = counts.get(0, 0)
        fraud_count = counts.get(1, 0)
        total = legit_count + fraud_count
        print("\nClass Distribution:")
        print(f"  Legitimate (0): {legit_count} ({legit_count/total*100:.2f}%)")
        print(f"  Fraudulent (1): {fraud_count} ({fraud_count/total*100:.2f}%)")
        print(f"  Imbalance Ratio (Legit/Fraud): {legit_count / max(1, fraud_count):.2f}x")
    else:
        print("\nWarning: 'fraudulent' label column not found in dataset!")

    # Check text fields null values
    text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    print("\nNull Values in Text Fields:")
    for col in text_fields:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            print(f"  {col}: {null_count} / {len(df)} nulls ({null_count/len(df)*100:.1f}%)")
        else:
            print(f"  {col}: Column missing!")
    print("="*40)

if __name__ == "__main__":
    main()
