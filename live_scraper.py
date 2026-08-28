# type: ignore
import requests
from bs4 import BeautifulSoup
import time
import random

class LiveScraper:
    _cached_mock_jobs = None

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        self.url = "https://remote.co/remote-jobs/writing/"

    def scrape_jobs(self, max_jobs=5):
        print(f"Attempting to scrape live jobs from {self.url}...")
        try:
            response = requests.get(self.url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                print(f"Warning: Received status code {response.status_code}. Using mock fallback.")
                return random.sample(self.get_mock_jobs(), min(max_jobs, len(self.get_mock_jobs())))
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remote.co layout: Jobs are usually under a container with class card-body or a list
            # Let's search for job links that have the pattern '/job/'
            job_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if "/job/" in href and href not in job_links:
                    # Filter out category landing pages or unrelated links
                    if not any(cat in href for cat in ['/job/remote-jobs', '/job/category', '/job/blog']):
                        # Remote.co links are sometimes relative, ensure absolute
                        if href.startswith('/'):
                            href = "https://remote.co" + href
                        job_links.append((a, href))

            if not job_links:
                print("Warning: No job links parsed from HTML structure. Using mock fallback.")
                return random.sample(self.get_mock_jobs(), min(max_jobs, len(self.get_mock_jobs())))

            print(f"Found {len(job_links)} potential job postings on the page. Processing top {max_jobs}...")
            
            jobs = []
            count = 0
            for a_tag, link in job_links:
                if count >= max_jobs:
                    break
                    
                try:
                    # Clean title and company from listing
                    # On Remote.co, title is often inside class "font-weight-bold" or inside the text of the link
                    title = a_tag.text.strip()
                    # Try to find parent/sibling text representing company
                    parent = a_tag.find_parent()
                    company = "Unknown Company"
                    if parent:
                        # Company is often adjacent text or has class 'co' or similar
                        co_elem = parent.find(class_='co') or parent.find('p')
                        if co_elem:
                            company = co_elem.text.strip().split('\n')[0]
                    
                    print(f"Scraping details for: {title} ({company})...")
                    time.sleep(1.0 + random.random()) # Polite delay
                    
                    detail_resp = requests.get(link, headers=self.headers, timeout=10)
                    if detail_resp.status_code == 200:
                        detail_soup = BeautifulSoup(detail_resp.content, 'html.parser')
                        
                        # Find job description body
                        # On Remote.co, this is often inside a div with class 'job_description' or 'job-description'
                        desc_div = detail_soup.find(class_='job_description') or detail_soup.find(class_='job-description')
                        
                        if desc_div:
                            description = desc_div.text.strip()
                        else:
                            # Fallback: get all paragraph text
                            paragraphs = detail_soup.find_all('p')
                            description = "\n".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
                        
                        # Set default values for other EMSCAD fields
                        jobs.append({
                            'title': title,
                            'company_profile': f"Information about {company} remote operations.",
                            'description': description,
                            'requirements': "Please refer to the job description details.",
                            'benefits': "Standard remote employment benefits apply.",
                            'is_mock': False,
                            'url': link
                        })
                        count += 1
                except Exception as e:
                    print(f"Error scraping details for link {link}: {e}")
                    continue
                    
            if not jobs:
                print("Warning: All individual detail pages failed. Using mock fallback.")
                return random.sample(self.get_mock_jobs(), min(max_jobs, len(self.get_mock_jobs())))
                
            return jobs
            
        except Exception as e:
            print(f"Network error or scraping failure: {e}. Using mock fallback.")
            return random.sample(self.get_mock_jobs(), min(max_jobs, len(self.get_mock_jobs())))

    def get_mock_jobs(self):
        if LiveScraper._cached_mock_jobs is not None:
            return LiveScraper._cached_mock_jobs

        print("Loading all fake and legitimate job postings from fake_job_postings.csv...")
        import os
        import csv
        mock_jobs = []
        csv_path = "fake_job_postings.csv"
        if not os.path.exists(csv_path):
            csv_path = "d:/ORF/fake_job_postings.csv"

        if os.path.exists(csv_path):
            try:
                with open(csv_path, encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mock_jobs.append({
                            'job_id': row.get('job_id'),
                            'title': row.get('title', 'Untitled'),
                            'company_profile': row.get('company_profile', ''),
                            'description': row.get('description', ''),
                            'requirements': row.get('requirements', ''),
                            'benefits': row.get('benefits', ''),
                            'is_mock': True,
                            'url': f"/mock-job/{row.get('job_id')}"
                        })
            except Exception as e:
                print(f"Error loading fake_job_postings.csv: {e}")

        # If loading failed or empty, fallback to a minimal static list
        if not mock_jobs:
            print("Warning: CSV loading yielded no jobs. Using minimal static fallback.")
            mock_jobs = [
                {
                    'job_id': "101",
                    'title': "Senior Content Writer (Remote)",
                    'company_profile': "We are a fully remote digital marketing agency founded in 2018. We help SaaS brands grow their search traffic and revenue through high-quality blog posts and content strategy.",
                    'description': "We are seeking a Senior Content Writer to create engaging articles, case studies, and ultimate guides. You will work closely with our editors to outline, write, and polish content that ranks well and delights readers. This is a full-time, contract role offering flexible working hours.",
                    'requirements': "A minimum of 5 years of professional writing experience, preferably in the B2B SaaS industry. Solid understanding of SEO principles. A portfolio showcasing published articles. Exceptional attention to grammar and style guide compliance.",
                    'benefits': "Competitive per-word rate or monthly retainer. 100% remote workspace. Flexible scheduling. Opportunities for promotion to editor or content manager.",
                    'is_mock': True,
                    'url': "/mock-job/101"
                },
                {
                    'job_id': "102",
                    'title': "Data Entry Clerk - Work From Home (Immediate Hire)",
                    'company_profile': "Apex Global Solutions Inc. is a leading logistics and outsourcing services provider operating worldwide.",
                    'description': "Earn easy money from the comfort of your home. We are hiring Data Entry Clerks to input records into our remote database. No experience is required. Anyone can apply. We provide full step-by-step training. You can work 2-3 hours per day and make over $3500 weekly.",
                    'requirements': "Must have a laptop or computer and a reliable internet connection. Basic typing speed. We conduct our interview process strictly via Telegram. To schedule an interview, please install Telegram and send a message to @apex_hr_manager to begin immediately.",
                    'benefits': "High daily pay. Weekly bonuses. Work from anywhere. Free software tools package.",
                    'is_mock': True,
                    'url': "/mock-job/102"
                }
            ]

        LiveScraper._cached_mock_jobs = mock_jobs
        return mock_jobs

if __name__ == "__main__":
    scraper = LiveScraper()
    jobs = scraper.scrape_jobs(max_jobs=2)
    print(f"\nSuccessfully fetched {len(jobs)} jobs:")
    for j in jobs:
        print(f"- {j['title']} (Mock: {j.get('is_mock', False)})")
