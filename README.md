# 🚀 Skill Gap Recommendation Engine

> **An end-to-end ML-powered career intelligence platform** that analyzes your resume, verifies your skills through AI-generated assessments, and delivers a personalized career roadmap based on real job market trends.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Project Type** | Machine Learning + NLP + Web Application |
| **Target** | Suggest learning paths based on job market trends |
| **ML Models** | KNN (NearestNeighbors) + Linear Regression |
| **AI Engine** | Groq API (LLaMA 3.3 70B) |
| **UI Framework** | Gradio |
| **Database** | PostgreSQL |
| **Language** | Python 3.10+ |

---

## 🎯 Problem Statement

Fresh graduates and working professionals often don't know:
- Which technical skills are in demand right now
- How their current skills compare to job requirements
- What exact skills they need to learn to get their target role
- In what order they should learn missing skills

This project solves all four problems in one tool.

---

## 🤖 Machine Learning Models

### Model 1 — KNN Job Matcher
```
Algorithm  : KNeighborsClassifier (sklearn)
Metric     : Cosine Similarity
k          : 10 neighbors
Training   : 493 job roles × 80 skill features
Output     : Top 3 job matches with confidence %, shared skills, gap skills
```

**Why KNN over plain cosine similarity:**
- sklearn's BruteForce index searches all 493 job vectors simultaneously
- Returns calibrated distance-based confidence scores (0-100%)
- Provides explainability — which skills matched and which are missing
- k=10 gives broader search space for more robust recommendations

### Model 2 — Skill Demand Forecaster
```
Algorithm  : LinearRegression (sklearn) — one model per skill
Training   : 8 simulated quarterly data points per skill
Features   : Time (quarter index 0-7)
Target     : Skill demand count
Output     : Rising / Stable / Declining + growth % + R² score
```

**Why Linear Regression:**
- Slope coefficient directly tells if demand is increasing or falling
- R² score validates how well the trend fits the data
- Simple and interpretable — ideal for explainable AI
- Fitted independently on each of 690 skills

---

## ✨ Features

### 📄 Phase 1 — Core Intelligence
- **Resume PDF Parsing** — Extracts text and detects technical skills using NLP pattern matching
- **AI Quiz Generation** — Groq LLaMA generates 20-25 personalized MCQ questions from your skills
- **KNN Job Matching** — Top 3 career matches with confidence scores
- **Job Trend Analysis** — Linear Regression forecasts which skills are rising vs declining
- **Learning Path Generator** — Week-by-week study plan with free resources for each missing skill

### 📊 Phase 2 — User Experience
- **User Dashboard** — Assessment history, skill scores over time
- **PDF Report Export** — Downloadable career analysis report (ReportLab)
- **Skill Progress Chart** — Visual bar chart of verified skill levels (Matplotlib)

### 🤖 Phase 3 — Advanced AI Tools
- **Resume Improvement Suggestions** — AI scores your resume 1-10 with section-by-section fixes
- **Job Description Matcher** — Paste any JD and see your match score instantly
- **Salary Insights** — Entry/Mid/Senior salary ranges for your top matched roles

### 📜 Certificate Management
- Add up to 5 certificates with name, skills covered, and PDF upload
- Certificate PDFs are parsed and skills extracted automatically

---

## 🗂️ Project Structure

```
skill-gap-recommendation-engine/
│
├── src/
│   ├── gradio_app.py          # Main UI — all 5 tabs (Gradio)
│   ├── ml_engine.py           # KNN + Linear Regression models
│   ├── assessment_engine.py   # Groq AI quiz generator
│   ├── resume_parser.py       # PDF text extraction + skill detection
│   ├── database.py            # PostgreSQL — users, skills, history
│   ├── report_generator.py    # PDF report builder (ReportLab)
│   └── recommendation_engine.py  # Legacy CLI recommendation engine
│
├── data/
│   ├── job_skill_matrix.csv   # 493 jobs × 80 skills (binary matrix)
│   ├── skill_frequency.csv    # 690 skills with market demand counts
│   ├── IT_Job_Roles_Skills.csv# Job descriptions + certifications
│   ├── cleaned_jobs_dataset.csv
│   └── top_skills.csv         # Master skill vocabulary
│
├── models/
│   ├── knn_model.pkl          # Trained KNN model (auto-generated)
│   └── regression_forecasts.pkl # Trained regression models (auto-generated)
│
├── notebooks/                 # EDA and experimentation
├── Sample_Resume.pdf          # Test resume
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/skill-gap-recommendation-engine.git
cd skill-gap-recommendation-engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL
```sql
-- In pgAdmin or psql:
CREATE DATABASE skill_gap_db;
```

### 4. Create `.env` File
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
DB_PASSWORD=your_postgres_password
```

> **Get your free Groq API key:** https://console.groq.com

### 5. Create Models Folder
```bash
mkdir models
```

### 6. Run the App
```bash
python src/gradio_app.py
```

Open your browser at: **http://localhost:7860**

> ✅ ML models train automatically on first run (~3 seconds)

---

## 🖥️ How to Use

```
Step 1 → Tab 1: Upload resume PDF + add certificates → Parse Resume
Step 2 → Tab 2: Answer the AI-generated quiz questions → Submit
Step 3 → Tab 3: View Top 3 job matches, skill scores, trend forecast, learning path + download PDF
Step 4 → Tab 4: Enter email → Load Dashboard to see history and progress chart
Step 5 → Tab 5: AI Tools → Resume suggestions, JD matcher, Salary insights
```

---

## 📊 Dataset

| File | Rows | Description |
|------|------|-------------|
| `job_skill_matrix.csv` | 493 | Binary matrix: job × skill (1=required, 0=not required) |
| `skill_frequency.csv` | 690 | Skill demand count across all job roles |
| `IT_Job_Roles_Skills.csv` | 493 | Job descriptions, required skills, certifications |
| `top_skills.csv` | 80 | Master vocabulary for skill extraction from resumes |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 4.x |
| ML | scikit-learn (KNN, LinearRegression) |
| AI | Groq API — LLaMA 3.3 70B |
| NLP | PyPDF2, regex pattern matching |
| Database | PostgreSQL + psycopg2 |
| PDF Report | ReportLab |
| Charts | Matplotlib |
| Data | Pandas, NumPy |
| Environment | python-dotenv |

---

## 🔑 API Keys Required

| Key | Purpose | Get It |
|-----|---------|--------|
| `GROQ_API_KEY` | Quiz generation + AI tools | https://console.groq.com (free) |
| `GEMINI_API_KEY` | Optional fallback | https://aistudio.google.com |

---

## 👨‍💻 Author

Built as a capstone ML project demonstrating:
- End-to-end ML pipeline (data → model → deployment)
- Real-world NLP (resume parsing, skill extraction)
- Practical AI integration (LLM-powered assessments)
- Full-stack Python web application

---

## 📄 License

This project is for educational purposes.