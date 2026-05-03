# 🎯 Career Gap-Filler Advisor

> An AI-powered tool that compares your resume against a target job description and generates a **personalized 7-day learning roadmap** to bridge the skills gap — optionally delivered straight to your inbox.

---

## 🚀 Live Demo

2 min Live Demo Link- https://www.loom.com/share/0fb33082dff34b3694bfb970f77ea931

---

## 💡 Problem Statement

Job seekers upload their resume to job portals but never know **exactly** which skills are holding them back. Generic resume reviewers give vague feedback. This tool goes further — it:

1. Reads your resume
2. Compares it against a specific JD
3. Identifies the **exact missing skills/keywords**
4. Creates a **7-day, day-by-day learning plan** with free resources
5. Emails the roadmap directly to you

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Backend | Python + Flask |
| AI / LLM | Groq API (`llama3-70b-8192`) — **free** |
| PDF Parsing | PyPDF2 |
| JD Scraping | BeautifulSoup + Requests |
| Email | Gmail SMTP (App Password) |
| Frontend | Vanilla HTML/CSS/JS (no framework) |

---

## 🧠 AI Prompts Used

### Core Gap Analysis Prompt

```
You are an expert career coach and skills gap analyst.

Given:
1. A candidate's RESUME
2. A JOB DESCRIPTION they are targeting

Tasks:
1. Identify TOP 5-7 missing skills/keywords
2. Generate a personalized 7-DAY learning roadmap

Return as JSON:
{
  "job_title": "...",
  "candidate_name": "...",
  "missing_skills": [
    {"skill": "...", "importance": "Critical/High/Medium", "reason": "..."}
  ],
  "roadmap": [
    {
      "day": 1,
      "focus": "...",
      "tasks": ["..."],
      "resources": [{"title": "...", "url": "...", "type": "Course/Article/Video"}],
      "goal": "..."
    }
  ],
  "summary": "..."
}
```

**Why this works**: The structured JSON output forces the model to be specific and actionable. Using `llama3-70b` on Groq gives high quality with zero cost.

---

## 📋 Sample Input / Output

### Sample 1 — Data Analyst vs SDE Role

**Input Resume Skills**: Python, Excel, SQL, Power BI  
**Target JD**: Full Stack Developer — React, Node.js, REST APIs, Docker

**Output (partial)**:
```json
{
  "missing_skills": [
    {"skill": "React.js", "importance": "Critical", "reason": "Primary frontend framework listed in JD"},
    {"skill": "Node.js", "importance": "Critical", "reason": "Backend runtime, central to role"},
    {"skill": "Docker", "importance": "High", "reason": "Containerization required for deployment"}
  ],
  "roadmap": [
    {
      "day": 1, "focus": "React Fundamentals",
      "tasks": ["Install Node.js and create-react-app", "Build a counter component", "Learn JSX syntax"],
      "resources": [{"title": "React Docs", "url": "https://react.dev", "type": "Course"}],
      "goal": "Understand components and state"
    }
  ]
}
```

### Sample 2 — CS Fresher vs Product Manager Role

**Input Resume Skills**: C++, Java, Data Structures  
**Target JD**: Associate PM — Roadmapping, Stakeholder Management, SQL, Figma, A/B Testing

**Output**: Roadmap covering SQL on Day 1-2, Figma basics Day 3, A/B testing fundamentals Day 4-5, PM frameworks Day 6-7.

### Sample 3 — Finance Grad vs Data Analyst

**Input Resume Skills**: Financial Modeling, Excel, Bloomberg, Valuation  
**Target JD**: Data Analyst — Python, pandas, Tableau, Statistics

**Output**: Python crash course Day 1-2, pandas + data cleaning Day 3-4, Tableau Day 5-6, Stats concepts Day 7.

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/career-gap-filler.git
cd career-gap-filler
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your keys:
# GROQ_API_KEY — get free at https://console.groq.com
# GMAIL_ADDRESS + GMAIL_APP_PASSWORD — optional, for email feature
```

### 4. Run the app
```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 📁 File Structure

```
career-gap-filler/
├── app.py                  # Flask backend + Groq AI + Gmail logic
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── templates/
│   └── index.html          # Full frontend (single file)
├── uploads/                # Temp PDF storage (auto-cleaned)
└── README.md
```

---

## 🔑 Getting Your Free Keys

**Groq API Key (Required)**:
1. Go to https://console.groq.com
2. Sign up (free)
3. Go to API Keys → Create Key
4. Paste in `.env`

**Gmail App Password (Optional, for email)**:
1. Enable 2FA on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Create an App Password for "Mail"
4. Paste the 16-char password in `.env`

---

## 🚧 Error Handling

- **Invalid PDF**: Returns clear error message, doesn't crash
- **URL scraping fails**: Falls back to error with instructions to paste JD text instead
- **AI returns malformed JSON**: Caught with `json.JSONDecodeError`, returns user-friendly error
- **Email fails**: Job still completes, email failure reported separately (non-blocking)
- **Empty inputs**: Validated on both frontend and backend

---

## ✨ Lessons Learned

1. **Prompt engineering for JSON**: Adding "Return ONLY the JSON, no markdown" and stripping code fences with regex was essential — LLMs often wrap JSON in ` ```json ``` `.
2. **Groq speed**: Groq's `llama3-70b` is shockingly fast (~2-3s for this prompt) vs OpenAI. Perfect for real-time web apps.
3. **URL scraping is unreliable**: LinkedIn and most job boards block scrapers. Added a "paste JD text" fallback as primary path.
4. **Token limits**: Job descriptions can be very long. Truncating to 2000 chars and resumes to 3000 chars kept quality high without hitting limits.

---

## 🔮 Future Improvements

- [ ] Add RAG: store past roadmaps in a vector DB, retrieve similar plans for faster generation
- [ ] GitHub Actions integration: auto-commit roadmap as a `.md` file to user's repo
- [ ] Skill tracking: checkboxes in the roadmap, stored in localStorage
- [ ] LinkedIn job URL parsing with Playwright/Selenium
- [ ] Multi-resume comparison (current vs target profile over time)
- [ ] Slack/WhatsApp notification instead of just email

---

## 📊 Prompt Comparison

| Prompt Style | Output Quality |
|---|---|
| "Review this resume vs JD" | Vague, unstructured feedback |
| "List missing skills as JSON" | Structured but no roadmap |
| "Identify gaps AND generate 7-day roadmap as JSON with resources" | ✅ Best — actionable, structured, complete |

The key insight: **the more constrained and output-format-specific the prompt, the more useful the result**.

---

*Built for the AI Projects competition. Start early, finish strong. 🚀*