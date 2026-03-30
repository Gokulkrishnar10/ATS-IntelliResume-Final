# ATS-IntelliResume: AI-Powered Career Optimization System

**ATS-IntelliResume** is a final-year project designed to bridge the structural and vocabulary gap between candidate resumes and modern Applicant Tracking Systems (ATS). It transforms the resume optimization process from "guesswork" into a fact-preserving engineering lifecycle.

## 🚀 LIVE DEMO
[Deploying to Streamlit Cloud...]

## 🛠️ Core Features
- **10-Phase Pipeline:** A highly modular execution engine—from environment validation to final LaTeX typesetting.
- **Three-Layer Hybrid Matching:**
  1. **Deterministic:** Exact keyword intersection.
  2. **Semantic:** Vector-based synonym discovery using `SentenceTransformers`.
  3. **Abstract:** LLM-powered inference for implicit skill detection.
- **Quantitative ATS Scoring:** Generates a composite score ($S_{ATS}$) based on keyword density, formatting, and completeness.
- **Human-in-the-Loop (HITL):** An interactive UI allows candidates to approve or reject AI-generated suggestions before the final export.
- **Phase 10 LaTeX Export:** Automatically generates 5 professional, ATS-safe LaTeX templates with automated syntax verification.

## 🏗️ Technical Architecture
The system is built using:
- **Core:** Python, Streamlit
- **Brain:** Groq (LLaMA-3.1-70b)
- **Embeddings:** `all-MiniLM-L6-v2`
- **Doc Processing:** `PyPDF2`, `python-docx`, `reportlab`

## 📦 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Gokulkrishnar10/ATS-IntelliResume-Final.git
