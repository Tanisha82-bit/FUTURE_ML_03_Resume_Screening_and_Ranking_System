import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import docx

# --- Page Configuration ---
st.set_page_config(
    page_title="ATSPulse | AI Applicant Tracking System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Dark Navy Premium SaaS Theme) ---
st.markdown("""
<style>
    /* Dark Navy Palette Definitions */
    :root {
        --bg-main: #0B1120;
        --sidebar-bg: #111B2E;
        --card-bg: #17243A;
        --primary-blue: #579BFF;
        --accent-cyan: #43D9CE;
        --text-main: #F4F7FC;
        --text-sub: #A3B1C6;
    }
    
    /* Main App Background */
    .stApp {
        background-color: #0B1120;
        color: #F4F7FC;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111B2E !important;
        border-right: 1px solid #1E2D4A;
    }
    
    /* Card Container */
    div.css-card {
        background-color: #17243A;
        border: 1px solid #243552;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        color: #579BFF !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #A3B1C6 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Skill Dictionary Definition ---
SKILL_GROUPS = {
    'programming': ['python', 'r', 'java', 'c++', 'c#', 'javascript', 'typescript', 'html', 'css', 'sql'],
    'data_analysis': ['excel', 'advanced excel', 'power bi', 'tableau', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'statistics', 'data analysis', 'data visualization'],
    'machine_learning': ['machine learning', 'deep learning', 'scikit-learn', 'tensorflow', 'pytorch', 'natural language processing', 'nlp', 'computer vision', 'model evaluation'],
    'databases_and_tools': ['mysql', 'postgresql', 'mongodb', 'git', 'github', 'jupyter', 'jupyter notebook', 'vs code', 'streamlit'],
    'cloud_and_big_data': ['aws', 'azure', 'google cloud', 'spark', 'hadoop']
}

ALL_SKILLS = sorted({skill for group in SKILL_GROUPS.values() for skill in group}, key=len, reverse=True)

# --- Document Parsing Helper ---
def extract_text_from_file(uploaded_file):
    """Extract raw text from PDF, DOCX, or TXT file."""
    filename = uploaded_file.name.lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        elif filename.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error parsing {uploaded_file.name}: {e}")
    return text

# --- Text Cleaning & Skill Extraction ---
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r'[^a-z0-9+#./\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text, skill_list=None):
    cleaned = clean_text(text)
    skill_list = skill_list or ALL_SKILLS
    found = []
    for skill in skill_list:
        escaped = re.escape(skill.lower()).replace(r'\ ', r'\s+')
        pattern = rf'(?<![a-z0-9+#]){escaped}(?![a-z0-9+#])'
        if re.search(pattern, cleaned):
            found.append(skill)
    return sorted(set(found))

# --- Session State Initialization ---
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = None
if 'job_title' not in st.session_state:
    st.session_state['job_title'] = "Entry-Level Data Analyst"
if 'job_desc' not in st.session_state:
    st.session_state['job_desc'] = "We are looking for an entry-level Data Analyst. Required skills: Python, SQL, Excel, Power BI, Statistics, Pandas, Numpy, Machine Learning."
if 'req_skills' not in st.session_state:
    st.session_state['req_skills'] = "python, sql, excel, power bi, statistics, machine learning, pandas, numpy"

# --- Sidebar Navigation & Controls ---
with st.sidebar:
    st.markdown("## 💼 **ATSPulse AI**")
    st.caption("AI Applicant Tracking & Screening")
    st.divider()
    
    nav_choice = st.radio(
        "Navigation",
        ["📊 Overview Dashboard", "📌 Job Requirement", "📝 Single Screening", "📁 Batch Screening", "📈 Analytics & Insights", "⚙️ Model Settings"],
        index=0
    )
    
    st.divider()
    st.markdown("### ⚙️ Scoring Weights")
    skill_weight = st.slider("Skill Match Weight", 0.0, 1.0, 0.70, 0.05)
    similarity_weight = round(1.0 - skill_weight, 2)
    st.markdown(f"**TF-IDF Weight:** `{similarity_weight}`")
    
    st.divider()
    st.caption("Internship Task 03 • ML Recruitment SaaS Prototype")

# --- HEADER SECTION ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("AI-Powered Resume Screening & Ranking System")
    st.caption(f"Active Job Profile: **{st.session_state['job_title']}**")

# ==========================================
# PAGE 1: OVERVIEW DASHBOARD
# ==========================================
if nav_choice == "📊 Overview Dashboard":
    st.subheader("System Overview & Candidate Metrics")
    
    if st.session_state['batch_results'] is not None:
        res_df = st.session_state['batch_results']
        total_screened = len(res_df)
        avg_score = res_df['final_match_score'].mean()
        strong_matches = len(res_df[res_df['final_match_score'] >= 50])
        skills_count = len(ALL_SKILLS)
    else:
        total_screened = 0
        avg_score = 0.0
        strong_matches = 0
        skills_count = len(ALL_SKILLS)
        
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates Screened", f"{total_screened}")
    m2.metric("Average Match Score", f"{avg_score:.1f}%")
    m3.metric("Strong Matches (≥50%)", f"{strong_matches}")
    m4.metric("Skill Vocab Size", f"{skills_count}")
    
    st.divider()
    
    if st.session_state['batch_results'] is not None:
        st.markdown("### 🏆 Top Ranked Candidates Leaderboard")
        disp_cols = ['rank', 'candidate_id', 'final_match_score', 'skill_match_score', 'similarity_score', 'matched_skills', 'missing_skills']
        st.dataframe(st.session_state['batch_results'][disp_cols].head(10), use_container_width=True)
    else:
        st.info("👋 Welcome to **ATSPulse**. Start by configuring the **Job Requirement** or upload candidate resumes under **Single / Batch Screening**.")
        
        col_box1, col_box2 = st.columns(2)
        with col_box1:
            st.markdown("""
            #### 🎯 Quick Workflow Guide:
            1. **Job Requirement:** Set the target job role, paste JD, & edit skills.
            2. **Single Screening:** Test individual PDF, DOCX, or TXT resumes.
            3. **Batch Screening:** Upload multiple resume files or CSV datasets.
            4. **Analytics:** View score distributions, gap analysis, and charts.
            """)
        with col_box2:
            st.markdown("""
            #### ⚙️ Hybrid ML Engine:
            * **70% Skill Dictionary Match:** Boundary-aware exact keyword matching.
            * **30% TF-IDF Cosine Similarity:** Semantic text similarity calculation.
            * **Configurable:** Adjust weights in real-time from the sidebar.
            """)

# ==========================================
# PAGE 2: JOB REQUIREMENT SETUP
# ==========================================
elif nav_choice == "📌 Job Requirement":
    st.subheader("Target Job Profile Setup")
    st.markdown("Define the job description and required skill keywords used for candidate evaluation.")
    
    with st.form("job_setup_form"):
        j_title = st.text_input("Job Title", value=st.session_state['job_title'])
        j_desc = st.text_area("Job Description", height=200, value=st.session_state['job_desc'])
        
        extracted_from_desc = extract_skills(j_desc)
        suggested_skills = ", ".join(extracted_from_desc) if extracted_from_desc else st.session_state['req_skills']
        
        j_skills = st.text_input("Required Skills (comma-separated)", value=suggested_skills)
        
        submit_job = st.form_submit_button("Save Job Profile", type="primary")
        if submit_job:
            st.session_state['job_title'] = j_title
            st.session_state['job_desc'] = j_desc
            st.session_state['req_skills'] = j_skills
            st.success("Job Requirement setup saved successfully!")
            
    st.divider()
    st.markdown("### 🔍 Extracted Skill Chips")
    skill_chips = [s.strip() for s in st.session_state['req_skills'].split(",") if s.strip()]
    st.write(" ".join([f"`{chip}`" for chip in skill_chips]))

# ==========================================
# PAGE 3: SINGLE RESUME SCREENING
# ==========================================
elif nav_choice == "📝 Single Screening":
    st.subheader("Single Resume Analysis")
    st.markdown("Upload candidate resume in **PDF, DOCX, or TXT** format, or paste text directly.")
    
    tab_file, tab_text = st.tabs(["📄 Upload Resume File (PDF / DOCX / TXT)", "✍️ Paste Resume Text"])
    
    resume_text = ""
    candidate_name_file = "Single Candidate"
    
    with tab_file:
        uploaded_res = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])
        if uploaded_res is not None:
            resume_text = extract_text_from_file(uploaded_res)
            candidate_name_file = uploaded_res.name
            st.success(f"Parsed {uploaded_res.name} successfully ({len(resume_text)} characters extracted).")
            
    with tab_text:
        pasted_text = st.text_area("Paste Candidate Resume Text", height=250, placeholder="Paste resume contents here...")
        if pasted_text.strip():
            resume_text = pasted_text
            
    if st.button("Screen Resume Now", type="primary"):
        if not resume_text.strip():
            st.warning("Please upload a resume file or paste resume text first.")
        else:
            req_list = [s.strip().lower() for s in st.session_state['req_skills'].split(",") if s.strip()]
            cand_skills = extract_skills(resume_text)
            matched = sorted(list(set(cand_skills).intersection(set(req_list))))
            missing = sorted(list(set(req_list) - set(cand_skills)))
            
            skill_score = (len(matched) / len(req_list) * 100) if req_list else 0.0
            
            cleaned_jd = clean_text(st.session_state['job_desc'])
            cleaned_res = clean_text(resume_text)
            
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
            tfidf_mat = vectorizer.fit_transform([cleaned_jd, cleaned_res])
            sim_score = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0] * 100
            
            final_score = (skill_weight * skill_score) + (similarity_weight * sim_score)
            
            st.divider()
            st.markdown(f"### 🎯 Evaluation Results: `{candidate_name_file}`")
            
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Final Match Score", f"{final_score:.1f}%")
            res_c2.metric("Skill Match Score", f"{skill_score:.1f}%")
            res_c3.metric("TF-IDF Similarity", f"{sim_score:.1f}%")
            
            col_m, col_g = st.columns(2)
            with col_m:
                st.markdown("#### ✅ Matched Skills")
                st.write(" ".join([f"`{s}`" for s in matched]) if matched else "No direct matches found.")
            with col_g:
                st.markdown("#### ❌ Missing Skills")
                st.write(" ".join([f"`{s}`" for s in missing]) if missing else "No skill gap detected!")

# ==========================================
# PAGE 4: BATCH CANDIDATE SCREENING
# ==========================================
elif nav_choice == "📁 Batch Screening":
    st.subheader("Batch Candidate Screening & Ranking")
    st.markdown("Screen candidate pools via **multiple resume files (PDF / DOCX / TXT)** or a **CSV dataset**.")
    
    batch_mode = st.radio("Choose Batch Source", ["Multiple File Upload (PDF / DOCX / TXT)", "CSV Dataset Upload"], horizontal=True)
    
    candidates_data = []
    
    if batch_mode == "Multiple File Upload (PDF / DOCX / TXT)":
        files = st.file_uploader("Upload Multiple Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if files:
            for f in files:
                txt = extract_text_from_file(f)
                if txt.strip():
                    candidates_data.append({'candidate_id': f.name, 'resume_text': txt})
            st.info(f"Loaded {len(candidates_data)} resume files.")
            
    else:
        csv_file = st.file_uploader("Upload CSV File", type=["csv"])
        if csv_file is not None:
            df_input = pd.read_csv(csv_file)
            st.write("Uploaded CSV Preview:", df_input.head(3))
            
            text_candidates = [c for c in df_input.columns if any(k in c.lower() for k in ['resume', 'text', 'description', 'ticket description'])]
            selected_col = st.selectbox("Select Resume Text Column", text_candidates if text_candidates else df_input.columns)
            
            for idx, row in df_input.iterrows():
                val = str(row[selected_col])
                if val.strip():
                    cand_name = str(row.get('Customer Name', f"CAND_{idx+1:04d}"))
                    candidates_data.append({'candidate_id': cand_name, 'resume_text': val})
                    
    if st.button("Run Batch Candidate Screening", type="primary"):
        if not candidates_data:
            st.warning("Please upload candidate files or a valid CSV first.")
        else:
            req_list = [s.strip().lower() for s in st.session_state['req_skills'].split(",") if s.strip()]
            
            b_df = pd.DataFrame(candidates_data)
            b_df['clean_resume'] = b_df['resume_text'].apply(clean_text)
            b_df['candidate_skills'] = b_df['resume_text'].apply(extract_skills)
            b_df['matched_skills'] = b_df['candidate_skills'].apply(lambda sk: sorted(list(set(sk).intersection(set(req_list)))))
            b_df['missing_skills'] = b_df['candidate_skills'].apply(lambda sk: sorted(list(set(req_list) - set(sk))))
            b_df['skill_match_score'] = b_df['matched_skills'].apply(lambda sk: (len(sk)/len(req_list)*100) if req_list else 0.0)
            
            cleaned_jd = clean_text(st.session_state['job_desc'])
            docs = [cleaned_jd] + b_df['clean_resume'].tolist()
            
            b_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
            b_tfidf = b_vectorizer.fit_transform(docs)
            b_sim = cosine_similarity(b_tfidf[0:1], b_tfidf[1:]).flatten() * 100
            
            b_df['similarity_score'] = b_sim
            b_df['final_match_score'] = (skill_weight * b_df['skill_match_score']) + (similarity_weight * b_df['similarity_score'])
            
            results_df = b_df.sort_values(by=['final_match_score', 'skill_match_score'], ascending=False).reset_index(drop=True)
            results_df.insert(0, 'rank', range(1, len(results_df) + 1))
            
            st.session_state['batch_results'] = results_df
            st.success("Batch screening completed successfully!")
            
            disp_cols = ['rank', 'candidate_id', 'final_match_score', 'skill_match_score', 'similarity_score', 'matched_skills', 'missing_skills']
            st.dataframe(results_df[disp_cols], use_container_width=True)
            
            csv_export = results_df[disp_cols].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Results CSV", data=csv_export, file_name="candidate_ranking_results.csv", mime="text/csv")

# ==========================================
# PAGE 5: ANALYTICS & INSIGHTS
# ==========================================
elif nav_choice == "📈 Analytics & Insights":
    st.subheader("Recruitment & Skill Gap Analytics")
    
    if st.session_state['batch_results'] is None:
        st.info("Run a batch screening analysis under 'Batch Screening' to unlock visual analytics.")
    else:
        res_df = st.session_state['batch_results']
        
        plt.style.use('dark_background')
        
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.markdown("### Top 10 Candidate Match Scores")
            top10 = res_df.head(10).sort_values(by='final_match_score')
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#17243A')
            ax.set_facecolor('#17243A')
            ax.barh(top10['candidate_id'], top10['final_match_score'], color='#579BFF')
            ax.set_xlabel("Final Match Score (%)", color='#A3B1C6')
            ax.set_ylabel("Candidate ID", color='#A3B1C6')
            ax.tick_params(colors='#A3B1C6')
            st.pyplot(fig)
            
        with c_a2:
            st.markdown("### Match Score Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor('#17243A')
            ax2.set_facecolor('#17243A')
            sns.histplot(res_df['final_match_score'], kde=True, color='#43D9CE', ax=ax2)
            ax2.set_xlabel("Final Match Score (%)", color='#A3B1C6')
            ax2.set_ylabel("Candidate Count", color='#A3B1C6')
            ax2.tick_params(colors='#A3B1C6')
            st.pyplot(fig2)
            
        st.divider()
        st.markdown("### Most Common Missing Skills (Skill Gap Frequency)")
        all_missing = [skill for sublist in res_df['missing_skills'] for skill in sublist]
        if all_missing:
            missing_counts = pd.Series(all_missing).value_counts().reset_index()
            missing_counts.columns = ['Skill', 'Missing Frequency']
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            fig3.patch.set_facecolor('#17243A')
            ax3.set_facecolor('#17243A')
            sns.barplot(data=missing_counts, x='Skill', y='Missing Frequency', palette='Reds_r', ax=ax3)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', color='#A3B1C6')
            ax3.tick_params(colors='#A3B1C6')
            st.pyplot(fig3)

# ==========================================
# PAGE 6: MODEL SETTINGS & DICTIONARY
# ==========================================
elif nav_choice == "⚙️ Model Settings":
    st.subheader("Model Configuration & Skill Dictionary")
    st.markdown("Inspect and customize the underlying explainable skill dictionary categories.")
    
    st.markdown(f"**Current Skill Match Weight:** `{skill_weight}` | **TF-IDF Weight:** `{similarity_weight}`")
    
    for domain, skills in SKILL_GROUPS.items():
        with st.expander(f"📁 Domain: {domain.upper()} ({len(skills)} skills)"):
            st.write(", ".join([f"`{s}`" for s in skills]))