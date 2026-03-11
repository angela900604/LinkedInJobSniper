import glob
import os
import smtplib
from typing import List, Optional
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import html
import json
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import re

# web clawing imports if needed
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import random

import pandas as pd
from dotenv import load_dotenv
from jobspy import scrape_jobs

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# read pdf
from pypdf import PdfReader


# Load environment variables
load_dotenv()

# Configuration
SEARCH_TERMS = [
    "UI Designer",
    "UX Designer",
    "Product Designer",
    "User Experience Designer",
    "UI/UX Designer",
    "User Interface Designer"
]

LOCATIONS = ["Canada", "Toronto, ON", "Vancouver, BC"]

RESULT_LIMIT = 40
HOURS_OLD = 72
PROXY_URL = os.getenv("PROXY_URL_LALA", None)
RESUME = os.getenv("RESUME_TEXT_LALA", None)
API_KEY = os.getenv("OPENAI_API_KEY_LALA")
BASE_URL = os.getenv("API_BASE")  # keep as requested
CRITERIA = os.getenv("CRITERIA_LALA", "")

print("RESUME raw:", repr(RESUME))
print("RESUME exists:", bool(RESUME))


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# Define the output data structure from AI
class JobEvaluation(BaseModel):
    """
    Structure for job evaluation output.
    """
    score: int = Field(description="A relevance score from 0 to 100 based on the resume match and job preferences.")
    reason: str = Field(description="A concise, one-sentence reason for the score.")


# AI model
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    api_key=API_KEY,
    base_url=BASE_URL,
)

# structured output
structured_llm = llm.with_structured_output(JobEvaluation)

# system template
system_template = """
[Context]
You are an expert tech career coach. Your goal is to evaluate how well a job description matches a candidate's resume and preferences.

[Objectives]
Return a score by the following criteria and also give a concise, one-sentence reason for the score.

[Criteria]
1. Skill Match (50%): How well do the required skills and technologies in the job description align with those listed on the resume? (Programming Languages, Frameworks, Tools, etc,)
"""

system_template += CRITERIA

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", """
    RESUME (Truncated):
    {resume}

    JOB TITLE: {title}
    JOB DESCRIPTION (Truncated):
    {description}

    Analyze the match. Be strict. If the tech stack is completely different, give a low score.
    """)
])

# Chain
evaluation_chain = prompt_template | structured_llm


# Read resume from Google Drive
def load_resume_from_google_drive() -> str:
    creds_json_str = os.getenv("GCP_CREDENTIALS_JSON")
    file_id = os.getenv("RESUME_FILE_ID")

    if not creds_json_str or not file_id:
        print("❌  Google Drive credentials or file ID not provided.")
        return None

    print("🔐  Loading resume from Google Drive...")

    try:
        creds_dict = json.loads(creds_json_str)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )

        service = build("drive", "v3", credentials=creds)

        request = service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"   ⬇️  Downloading... {int(status.progress() * 100)}%")

        file_io.seek(0)
        reader = PdfReader(file_io)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        print("✅  Resume loaded successfully from Google Drive.")
        return clean_text(text)

    except Exception as e:
        print(f"❌  Failed to load resume from Google Drive: {e}")
        return None


if not RESUME:
    RESUME = load_resume_from_google_drive()
else:
    RESUME = clean_text(RESUME)


# web crawling functions
def fetch_missing_description(url: str, proxies: dict = None) -> str:
    """
    If JobSpy cannot fetch description, try to fetch from job url directly.
    LinkedIn only for now.
    """
    print(f"   ⛑️  Attempting manual fetch for: {url}...")

    ua = UserAgent()
    headers = {
        "User-Agent": ua.random,
        "Accept-Language": "en-US,en;q=0.9",
        "Referrer": "https://www.google.com/"
    }

    try:
        time.sleep(random.uniform(2, 5))

        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            description_div = (
                soup.find("div", {"class": "show-more-less-html__markup"})
                or soup.find("div", {"class": "description__text"})
                or soup.find("div", {"class": "job-description"})
            )

            if description_div:
                text = description_div.get_text(separator="\n").strip()
                return clean_text(text)
            return clean_text(soup.get_text()[:5000])

        print(f"     ❌  Failed to fetch page, status code: {response.status_code}")
        return ""

    except Exception as e:
        print(f"     ❌  Exception during manual fetch: {str(e)}")
        return ""


# scrape jobs
def get_jobs_data(location: str, search_term: str) -> pd.DataFrame:
    """
    Scrape job listings by JobSpy.
    Add retry logic if needed.
    """
    proxies = [PROXY_URL] if PROXY_URL else None
    print(f"🕵️  CareerScout is searching for '{search_term}' in '{location}'...")
    print(f"🔌  Proxy: {proxies[0] if proxies else 'None'}")

    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            print(f"   🔄 Attempt {attempt} of {max_retries}...")
            jobs = scrape_jobs(
                site_name=["linkedin"],
                search_term=search_term,
                location=location,
                result_wanted=RESULT_LIMIT,
                hours_old=HOURS_OLD,
                proxies=proxies
            )

            print(f"✅  Scraped {len(jobs)} jobs.")
            return jobs

        except Exception as e:
            print(f"     ❌  Error on attempt {attempt}: {str(e)}")
            print(f"❌  Error during job scraping: {str(e)}")

            if attempt < max_retries:
                wait_time = random.uniform(3, 6)
                print(f"   ⏳ Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("All retry attempts failed. Exiting scraping process.")

    return pd.DataFrame()


def evaluate_job(title: str, description: str) -> dict:
    if not description or len(str(description)) < 50:
        return {"score": 0, "reason": "Job description too short or missing"}

    if not RESUME or len(clean_text(RESUME)) < 50:
        return {"score": 0, "reason": "Resume missing or invalid"}

    try:
        resume_text = clean_text(RESUME)[:3000]
        desc_text = clean_text(description)[:3000]
        title_text = clean_text(title or "")

        payload = {
            "resume": resume_text,
            "title": title_text,
            "description": desc_text
        }

        result: JobEvaluation = evaluation_chain.invoke(payload)

        if result is None:
            return {"score": 0, "reason": "AI returned None"}

        return {"score": result.score, "reason": clean_text(result.reason)}

    except Exception as e:
        print(f"⚠️ AI Evaluation Error for '{title}': {e}")
        return {"score": 0, "reason": "AI Error"}


def send_email(top_jobs: List[dict]):
    if not top_jobs:
        print("📭  No matching jobs to send.")
        return

    sender = clean_text(os.getenv("EMAIL_SENDER_LALA"))
    password = clean_text(os.getenv("EMAIL_PASSWORD_LALA"))
    receiver = clean_text(os.getenv("EMAIL_RECEIVER_LALA"))

    if not sender or not password or not receiver:
        print("❌  Missing email configuration.")
        return

    subject = clean_text(
        f"CareerScout: Top {len(top_jobs)} Jobs for {datetime.now().strftime('%Y-%m-%d')}"
    )

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #2c3e50;">CareerScout Daily Report</h2>
        <p>Found <b>{len(top_jobs)}</b> high-match positions for you today:</p>
        <table style="border-collapse: collapse; width: 100%; max-width: 800px;">
            <tr style="background-color: #f8f9fa; text-align: left;">
                <th style="padding: 10px; border-bottom: 2px solid #ddd;">Score</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd;">Title</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd;">Company</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd;">Why Match?</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd;">Action</th>
            </tr>
    """

    for job in top_jobs:
        score = job.get("score", 0)
        color = "#27ae60" if score >= 85 else "#d35400"

        title = html.escape(clean_text(job.get("title", "")))
        company = html.escape(clean_text(job.get("company", "")))
        reason = html.escape(clean_text(job.get("reason", "")))
        job_url = clean_text(job.get("job_url", ""))

        html_body += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: bold; color: {color};">
                    {score}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #eee;">{title}</td>
                <td style="padding: 10px; border-bottom: 1px solid #eee;">{company}</td>
                <td style="padding: 10px; border-bottom: 1px solid #eee; font-size: 14px; color: #555;">
                    {reason}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #eee;">
                    <a href="{job_url}" style="background-color: #007bff; color: white; padding: 5px 10px; text-decoration: none; border-radius: 4px; font-size: 12px;">Apply</a>
                </td>
            </tr>
        """

    html_body += """
        </table>
        <p style="margin-top: 20px; font-size: 12px; color: #888;">
            Powered by CareerScout-Agent using LangChain & Python.
        </p>
    </body>
    </html>
    """

    html_body = clean_text(html_body)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = str(Header(subject, "utf-8"))
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        print("DEBUG sender:", repr(sender))
        print("DEBUG receiver:", repr(receiver))
        print("DEBUG subject:", repr(subject))
        print("DEBUG has_xa0_sender:", "\xa0" in sender)
        print("DEBUG has_xa0_receiver:", "\xa0" in receiver)
        print("DEBUG has_xa0_subject:", "\xa0" in subject)
        print("DEBUG has_xa0_body:", "\xa0" in html_body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
        print(f"📧  Email sent successfully to {receiver}!")

    except Exception as e:
        print(f"❌  Email sending failed: {e}")


def main():
    # 1. Scraping
    df = pd.DataFrame()
    for location in LOCATIONS:
        for term in SEARCH_TERMS:
            jobs_df = get_jobs_data(location, term)
            if jobs_df is None or jobs_df.empty:
                continue

            jobs_df["search_term"] = term
            jobs_df["search_location"] = location
            df = pd.concat([df, jobs_df], ignore_index=True, sort=False)

    if df.empty:
        return

    # De-duplicate across keywords/locations.
    if "job_url" in df.columns:
        df = df.drop_duplicates(subset=["job_url"], keep="first")

    scored_jobs = []
    req_proxies = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

    # 2. Evaluation Loop
    print(f"🧠  Analyzing {len(df)} jobs with AI...")

    for _, row in df.iterrows():
        title = clean_text(row.get("title", "Unknown"))
        description = clean_text(row.get("description"))
        job_url = clean_text(row.get("job_url"))

        if not description or len(description) < 50:
            if job_url:
                description = clean_text(fetch_missing_description(job_url, proxies=req_proxies))

        if not description or len(description) < 50:
            print(f"   ⚠️  Skipping '{title}' due to insufficient description.")
            continue

        evaluation = evaluate_job(title, description)

        print()
        print(f"   📝 '{title}' scored {evaluation['score']}: {evaluation['reason']}")

        if evaluation["score"] >= 50:
            scored_jobs.append({
                "title": clean_text(title),
                "company": clean_text(row.get("company")),
                "job_url": clean_text(row.get("job_url")),
                "score": evaluation["score"],
                "reason": clean_text(evaluation["reason"])
            })

    # 3. Sorting & Sending
    scored_jobs.sort(key=lambda x: x["score"], reverse=True)
    top_20 = scored_jobs[:20]

    send_email(top_20)


if __name__ == "__main__":
    main()
