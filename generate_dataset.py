import pandas as pd
import random
import os

def generate_legit():
    titles = [
        "Software Engineer", "Data Analyst", "Marketing Manager", 
        "HR Manager", "Accountant", "Sales Representative",
        "Project Manager", "Customer Support Specialist"
    ]
    intros = [
        "We are looking for a dedicated professional to join our team.",
        "Join our fast-growing company and make an impact.",
        "An exciting opportunity for a seasoned professional.",
        "We are hiring! Come be a part of our innovative team."
    ]
    reqs = [
        "Requires 3+ years of experience in a similar role.",
        "A bachelor's degree in a related field is mandatory.",
        "Strong communication skills and attention to detail are required.",
        "Proficiency in Microsoft Office suite is preferred.",
        "Must be willing to work on-site in our downtown office."
    ]
    closings = [
        "Apply through our official company portal.",
        "Comprehensive health benefits and 401k provided.",
        "We value diversity and ensure equal opportunities.",
        "Please submit your resume and cover letter on our website."
    ]
    
    return f"{random.choice(titles)}\n\n{random.choice(intros)} {random.choice(reqs)} {random.choice(closings)}"

def generate_fraud():
    titles = [
        "Data Entry Clerk - WORK FROM HOME", "Easy Money Typist", 
        "Warehouse Assistant $$", "Immediate Hire - No Experience", 
        "Remote Assistant - High Pay"
    ]
    intros = [
        "Earn $5000 a week working just 2 hours a day from your laptop!",
        "Immediate start available! No experience needed, full training provided.",
        "Looking for people who want to make quick money working from home.",
        "Urgent hire for remote positions. Guaranteed high salary."
    ]
    red_flags = [
        "We require a remote interview via Telegram. Please install Telegram and contact @hr_recruiter.",
        "You must purchase your own equipment via our approved vendor using wire transfer.",
        "This role requires an upfront processing fee of $50 to secure your spot.",
        "Send us your banking details immediately so we can process your payroll setup.",
        "Payment is made via Western Union or cryptocurrency weekly."
    ]
    closings = [
        "Act fast, spots are limited!",
        "Text our hiring manager on WhatsApp to secure this position.",
        "No background checks required. Start tomorrow.",
        "Reply with your personal email and bank name to begin."
    ]
    
    return f"{random.choice(titles)}\n\n{random.choice(intros)} {random.choice(red_flags)} {random.choice(closings)}"

def main():
    random.seed(42)
    data = []
    
    # Generate 500 job postings (balanced dataset)
    for _ in range(250):
        data.append({"text": generate_legit(), "label": 0})
        data.append({"text": generate_fraud(), "label": 1})
        
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    out_path = os.path.join(os.path.dirname(__file__), "job_postings.csv")
    df.to_csv(out_path, index=False)
    print(f"Successfully generated 500 job postings in '{out_path}'")

if __name__ == "__main__":
    main()
