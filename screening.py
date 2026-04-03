
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===================== NLTK SETUP =====================
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except:
    pass

nltk.download("punkt")
nltk.download("stopwords")

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("🚀 Smart Resume Analyzer (ATS + NLP)")

# ===================== HELPERS =====================
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    try:
        words = word_tokenize(text)
    except:
        words = text.split()
    return " ".join(w for w in words if w not in stop_words)

# ===================== SKILL EXTRACTION =====================
def extract_skills(text):
    skills_db = [
        "python","java","sql","machine learning","deep learning",
        "nlp","data analysis","excel","tensorflow","pandas",
        "communication","teamwork","leadership","aws","docker"
    ]
    return {skill for skill in skills_db if skill in text}

# ===================== SMART SCORING =====================
def smart_score(resume, job):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume, job])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job)

    skill_match = len(resume_skills & job_skills) / (len(job_skills) + 1)

    final_score = (0.7 * sim + 0.3 * skill_match) * 100

    return round(final_score, 2), resume_skills, job_skills

# ===================== SECTION ANALYSIS =====================
def section_score(text):
    score = {
        "skills": 0,
        "projects": 0,
        "experience": 0
    }

    if "project" in text:
        score["projects"] = 1
    if "experience" in text or "internship" in text:
        score["experience"] = 1
    if any(skill in text for skill in ["python","java","sql"]):
        score["skills"] = 1

    return score

# ===================== SUMMARY =====================
def generate_summary(score, matched, missing):
    if score > 75:
        level = "strong"
    elif score > 50:
        level = "moderate"
    else:
        level = "weak"

    summary = f"Your resume shows a {level} match with the job description."

    if matched:
        summary += f" It includes relevant skills like {', '.join(list(matched)[:3])}."

    if missing:
        summary += f" However, it lacks focus on some important areas."

    return summary

# ===================== SUGGESTIONS =====================
def generate_suggestions(missing, section):
    suggestions = []

    if missing:
        suggestions.append("Add more job-relevant skills and keywords.")

    if section["projects"] == 0:
        suggestions.append("Include project experience.")

    if section["experience"] == 0:
        suggestions.append("Mention internships or work experience.")

    if section["skills"] == 0:
        suggestions.append("Highlight technical skills clearly.")

    if not suggestions:
        suggestions.append("Your resume looks strong.")

    return suggestions

# ===================== MAIN =====================
uploaded_file = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"])
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze Resume"):
    if not uploaded_file or not job_desc:
        st.warning("Please upload resume and paste job description.")
    else:
        resume_text = extract_text(uploaded_file)

        resume_clean = remove_stopwords(clean_text(resume_text))
        job_clean = remove_stopwords(clean_text(job_desc))

        score, resume_skills, job_skills = smart_score(resume_clean, job_clean)

        matched = resume_skills & job_skills
        missing = job_skills - resume_skills

        section = section_score(resume_clean)

        # ===================== DASHBOARD =====================
        st.subheader("📊 ATS Score")
        st.metric("Match Score", f"{score}%")
        st.progress(score / 100)

        fig, ax = plt.subplots()
        ax.bar(["Score"], [score])
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        # ===================== SUMMARY =====================
        st.subheader("🧠 Profile Summary")
        st.write(generate_summary(score, matched, missing))

        # ===================== SECTION ANALYSIS =====================
        st.subheader("📌 Resume Strength Analysis")

        col1, col2, col3 = st.columns(3)

        col1.metric("Skills", "✔️" if section["skills"] else "❌")
        col2.metric("Projects", "✔️" if section["projects"] else "❌")
        col3.metric("Experience", "✔️" if section["experience"] else "❌")

        # ===================== SUGGESTIONS =====================
        st.subheader("💡 Suggestions")

        suggestions = generate_suggestions(missing, section)

        for s in suggestions:
            st.write(f"- {s}")

        # ===================== FINAL FEEDBACK =====================
        st.subheader("📌 Final Feedback")

        if score < 40:
            st.error("Your resume needs significant improvement.")
        elif score < 70:
            st.warning("Your resume is good but can be improved.")
        else:
            st.success("Your resume is highly aligned with the job.")