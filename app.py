import os
import json
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from groq import Groq
import PyPDF2
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
    r"disregard\s+(your|all|the)\s+(instructions|rules|prompt)",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"forget\s+(everything|all|your\s+instructions)",
    r"return\s+only|output\s+only|respond\s+with\s+only",
    r"system\s*prompt",
]

def sanitize_input(text):
    """Remove potential prompt injection patterns from user input."""
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[removed]", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        return None, str(e)
    return text.strip(), None


BLOCKED_DOMAINS = ["linkedin.com", "indeed.com", "glassdoor.com", "naukri.com", "lever.co", "greenhouse.io"]

def extract_job_description(url_or_text):
    """Extract job description from a URL or raw text."""
    if url_or_text.startswith("http://") or url_or_text.startswith("https://"):
        from urllib.parse import urlparse
        domain = urlparse(url_or_text).netloc.lower().replace("www.", "")

        if any(blocked in domain for blocked in BLOCKED_DOMAINS):
            return None, f"{domain} blocks automated scraping. Please copy-paste the job description text directly instead."

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            resp = requests.get(url_or_text, headers=headers, timeout=10)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                tag.decompose()
            text = soup.get_text(separator='\n', strip=True)

            if len(text) < 100:
                return None, "The page returned very little text. It may require login or block scraping. Please paste the job description text instead."

            return text[:4000], None
        except requests.exceptions.HTTPError as e:
            return None, f"URL returned HTTP {e.response.status_code}. Please paste the job description text instead."
        except requests.exceptions.ConnectionError:
            return None, "Could not connect to the URL. Please check the link or paste the job description text instead."
        except requests.exceptions.Timeout:
            return None, "URL request timed out. Please paste the job description text instead."
        except Exception as e:
            return None, f"Could not fetch URL: {str(e)}. Please paste the job description text instead."
    else:
        return url_or_text.strip(), None


def analyze_gap_and_generate_roadmap(resume_text, job_description):
    """Use Groq to analyze the skill gap and generate a 7-day roadmap."""
    prompt = f"""You are an expert career coach and skills gap analyst.

I will give you:
1. A candidate's RESUME
2. A JOB DESCRIPTION they are targeting

Your task is to:
1. Identify ALL skills/keywords required in the job description
2. Determine which of those the candidate ALREADY HAS (matching skills)
3. Identify the TOP 5-7 MISSING skills/keywords
4. Calculate a match percentage (matching skills / total required skills * 100, rounded to nearest integer)
5. Generate a personalized 7-DAY learning roadmap to bridge the gaps

Return your response as a VALID JSON object with this exact structure:
{{
  "job_title": "extracted job title from JD",
  "candidate_name": "extracted from resume or 'Candidate'",
  "match_score": 62,
  "total_required_skills": 13,
  "matching_skills": [
    {{"skill": "skill name", "evidence": "where/how this appears in the resume"}}
  ],
  "missing_skills": [
    {{"skill": "skill name", "importance": "Critical/High/Medium", "reason": "why it matters for this role"}}
  ],
  "roadmap": [
    {{
      "day": 1,
      "focus": "main topic for the day",
      "tasks": ["task 1", "task 2", "task 3"],
      "resources": [
        {{"title": "resource name", "url": "https://free-resource-url.com", "type": "Course/Article/Video/Practice"}}
      ],
      "goal": "what you'll achieve by end of day"
    }}
  ],
  "summary": "2-3 sentence motivational summary of the plan"
}}

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Return ONLY the JSON, no markdown, no explanation."""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    return json.loads(chat_completion.choices[0].message.content)


def send_roadmap_email(to_email, candidate_name, roadmap_data):
    """Send the roadmap via Gmail SMTP."""
    sender_email = os.environ.get("GMAIL_ADDRESS")
    sender_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not sender_email or not sender_password:
        return False, "Email credentials not configured."

    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"🚀 Your 7-Day Career Roadmap for {roadmap_data.get('job_title', 'Your Target Role')}"
    msg['From'] = sender_email
    msg['To'] = to_email

    # Build HTML email
    days_html = ""
    for day in roadmap_data.get("roadmap", []):
        tasks_html = "".join([f"<li>{t}</li>" for t in day.get("tasks", [])])
        resources_html = "".join([
            f'<li><a href="{r.get("url","#")}" style="color:#6366f1;">{r.get("title")} [{r.get("type")}]</a></li>'
            for r in day.get("resources", [])
        ])
        days_html += f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:20px;margin-bottom:16px;border-left:4px solid #6366f1;">
          <h3 style="color:#a5b4fc;margin:0 0 8px;">Day {day['day']}: {day['focus']}</h3>
          <p style="color:#94a3b8;font-size:13px;margin:0 0 12px;">🎯 Goal: {day.get('goal','')}</p>
          <p style="color:#e2e8f0;font-weight:600;margin:8px 0 4px;">Tasks:</p>
          <ul style="color:#cbd5e1;margin:0 0 12px;">{tasks_html}</ul>
          <p style="color:#e2e8f0;font-weight:600;margin:8px 0 4px;">Resources:</p>
          <ul style="color:#cbd5e1;">{resources_html}</ul>
        </div>"""

    skills_html = "".join([
        f'<span style="background:#312e81;color:#a5b4fc;padding:4px 12px;border-radius:20px;font-size:13px;margin:4px;display:inline-block;">⚡ {s["skill"]} <em>({s["importance"]})</em></span>'
        for s in roadmap_data.get("missing_skills", [])
    ])

    html = f"""
    <html><body style="font-family:'Segoe UI',sans-serif;background:#0f0f1a;color:#e2e8f0;padding:24px;max-width:680px;margin:auto;">
      <div style="text-align:center;margin-bottom:32px;">
        <h1 style="color:#a5b4fc;font-size:28px;">🎯 Your Career Gap Roadmap</h1>
        <p style="color:#94a3b8;">Personalized for <strong style="color:#e2e8f0;">{candidate_name}</strong> → <strong style="color:#6366f1;">{roadmap_data.get('job_title','')}</strong></p>
      </div>
      <div style="background:#1e1e2e;border-radius:12px;padding:20px;margin-bottom:24px;">
        <h2 style="color:#f1f5f9;margin:0 0 12px;">📌 Skills to Bridge</h2>
        <div>{skills_html}</div>
      </div>
      <h2 style="color:#f1f5f9;">📅 Your 7-Day Plan</h2>
      {days_html}
      <div style="background:#1e293b;border-radius:12px;padding:20px;margin-top:24px;text-align:center;">
        <p style="color:#94a3b8;font-style:italic;">{roadmap_data.get('summary','')}</p>
        <p style="color:#475569;font-size:12px;margin-top:16px;">Generated by Career Gap-Filler Advisor 🚀</p>
      </div>
    </body></html>"""

    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get resume text
        resume_text = ""
        if 'resume_file' in request.files and request.files['resume_file'].filename:
            file = request.files['resume_file']
            filename = secure_filename(file.filename)
            if not filename.lower().endswith('.pdf'):
                return jsonify({"error": "Only PDF files are accepted."}), 400
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            resume_text, err = extract_text_from_pdf(file_path)
            os.remove(file_path)
            if err:
                return jsonify({"error": f"Could not read PDF: {err}"}), 400
        else:
            resume_text = request.form.get('resume_text', '').strip()

        if not resume_text:
            return jsonify({"error": "Please provide a resume (PDF or text)."}), 400

        resume_text = sanitize_input(resume_text)

        # Get job description
        job_input = request.form.get('job_input', '').strip()
        if not job_input:
            return jsonify({"error": "Please provide a job description or URL."}), 400

        job_description, err = extract_job_description(job_input)
        if err:
            return jsonify({"error": err}), 400

        job_description = sanitize_input(job_description)

        # Analyze and generate roadmap
        roadmap_data = analyze_gap_and_generate_roadmap(resume_text, job_description)

        # Optionally send email
        email = request.form.get('email', '').strip()
        email_status = None
        if email:
            candidate_name = roadmap_data.get('candidate_name', 'Candidate')
            success, msg = send_roadmap_email(email, candidate_name, roadmap_data)
            email_status = {"success": success, "message": msg}

        return jsonify({"roadmap": roadmap_data, "email_status": email_status})

    except json.JSONDecodeError as e:
        return jsonify({"error": f"AI returned invalid JSON. Try again. Detail: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)