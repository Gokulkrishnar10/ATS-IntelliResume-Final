# =============================================================================
# LATEX MODULE - Phase 10: LaTeX Resume Generator
# =============================================================================
# PURPOSE:
#   - Provides a Streamlit UI for selecting 5 LaTeX resume templates
#   - Fills selected template with resume data (from Phase 5 output)
#   - Sanitizes all content for LaTeX special character safety
#   - Uses Groq LLaMA-3 to verify and auto-fix the generated LaTeX
#
# HOW TO USE:
#   from latex_module import show_latex_phase
#   In Streamlit results section: show_latex_phase(optimized_resume, resume_struct, groq_client)
# =============================================================================

import re
import os
import streamlit as st
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LATEX SANITIZER
# ─────────────────────────────────────────────────────────────────────────────

def escape_latex(text: str) -> str:
    """
    Escapes LaTeX special characters in a given text string.
    This MUST be applied to every piece of user data before injecting into LaTeX.

    Characters handled:
        & → \&      (table column separator)
        % → \%      (LaTeX comment)
        $ → \$      (math mode)
        # → \#      (parameter reference)
        _ → \_      (subscript)
        { → \{      (group open)
        } → \}      (group close)
        ~ → \textasciitilde{}
        ^ → \textasciicircum{}
    """
    if not isinstance(text, str):
        text = str(text)

    # Order matters: escape backslash first to avoid double-escaping
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('~', r'\textasciitilde{}')
    text = text.replace('^', r'\textasciicircum{}')

    return text


def safe(text: str) -> str:
    """Shorthand for escape_latex. Use this everywhere when filling templates."""
    return escape_latex(text)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA EXTRACTOR
# Reads from Phase 5 + Phase 3 output dictionaries and normalizes them.
# ─────────────────────────────────────────────────────────────────────────────

def extract_resume_data(optimized_resume: dict, resume_struct: dict) -> dict:
    """
    Pulls all required fields from the Phase 5 output (optimized_resume)
    and Phase 3 output (resume_struct), normalizes them for LaTeX injection.

    Returns:
        data (dict): Flat dictionary of all values needed by any template.
    """
    personal = optimized_resume.get('personal_info', {})
    raw_personal = resume_struct.get('raw_json', {}).get('personal_info', {})

    name    = personal.get('name', '')      or raw_personal.get('name', 'CANDIDATE NAME')
    email   = personal.get('email', '')     or raw_personal.get('email', 'email@example.com')
    phone   = personal.get('phone', '')     or raw_personal.get('phone', '')
    linkedin = raw_personal.get('linkedin', '')
    github   = raw_personal.get('github', '')
    location = raw_personal.get('location', '')

    summary = optimized_resume.get('professional_summary', '')

    # Skills: categorized dict {category: [skill1, skill2, ...]}
    skills_categorized = optimized_resume.get('skills_section', {}).get('categorized', {})

    # Experience entries
    experience = optimized_resume.get('experience_section', {}).get('entries', [])

    # Project entries
    projects = optimized_resume.get('projects_section', {}).get('entries', [])

    # Education and certifications come from resume_struct (Phase 3)
    education = resume_struct.get('parsed_education', [])
    certifications = resume_struct.get('parsed_certifications', [])

    # Place and date for footer
    _loc_city = location.split(',')[-1].strip() if location else ''
    place = (
        st.session_state.get('enriched_contacts', {}).get('place', '') or
        _loc_city
    )
    today_date = datetime.now().strftime('%d.%m.%Y')

    return {
        'name': name,
        'email': email,
        'phone': phone,
        'linkedin': linkedin,
        'github': github,
        'location': location,
        'summary': summary,
        'skills_categorized': skills_categorized,
        'experience': experience,
        'projects': projects,
        'education': education,
        'certifications': certifications,
        'place': place,
        'today_date': today_date,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LaTeX SECTION BUILDERS
# These functions build individual LaTeX sections as strings.
# Each uses safe() to sanitize user data before inclusion.
# ─────────────────────────────────────────────────────────────────────────────

def build_skills_latex(skills_categorized: dict, style: str) -> str:
    """Builds the skills section in LaTeX, adapting to template style."""
    if not skills_categorized:
        return ""

    # Filter out non-skill items (institution names, overly long strings, etc.)
    _BAD_KW = {
        'university', 'institute', 'college', 'academy', 'school', 'vishwa',
        'vidyapeetham', 'amrita', 'deeplearning', 'coincent', 'cerebras',
        'coursera', 'udemy', 'edx', '.ai', 'pvt', 'ltd', 'inc.', 'corp',
    }
    def _ok(sk):
        s = sk.strip()
        if not s or len(s) > 45:
            return False
        sl = s.lower()
        if any(kw in sl for kw in _BAD_KW):
            return False
        words = s.split()
        if len(words) >= 3 and all(w and w[0].isupper() for w in words):
            return False
        return True

    if style == 'jakes_resume':
        lines = [r'\section{Technical Skills}', r'\begin{itemize}[leftmargin=0.15in, label={}]', r'\small{\item{']
        for category, skills in skills_categorized.items():
            skill_str = ', '.join([safe(s) for s in skills if _ok(s)])
            if not skill_str:
                continue
            lines.append(rf'    \textbf{{{safe(category)}}}{{{safe(":")}}} {skill_str} \\')
        lines.append(r'}}')
        lines.append(r'\end{itemize}')
        return '\n'.join(lines)

    elif style in ('simple_professional', 'engineering_minimal'):
        lines = [r'\section{Technical Skills}', r'\begin{itemize}[leftmargin=*, noitemsep]']
        for category, skills in skills_categorized.items():
            skill_str = ', '.join([safe(s) for s in skills if _ok(s)])
            if not skill_str:
                continue
            lines.append(rf'    \item \textbf{{{safe(category)}:}} {skill_str}')
        lines.append(r'\end{itemize}')
        return '\n'.join(lines)

    elif style == 'academic_cv':
        lines = [r'\section{Technical Skills}', r'\begin{itemize}[label=--]']
        for category, skills in skills_categorized.items():
            skill_str = ', '.join([safe(s) for s in skills if _ok(s)])
            if not skill_str:
                continue
            lines.append(rf'    \item \textbf{{{safe(category)}:}} {skill_str}.')
        lines.append(r'\end{itemize}')
        return '\n'.join(lines)

    elif style == 'moderncv':
        lines = []
        for category, skills in skills_categorized.items():
            skill_str = ', '.join([safe(s) for s in skills if _ok(s)])
            if not skill_str:
                continue
            lines.append(rf'\cvitem{{{safe(category)}}}{{{skill_str}}}')
        return '\n'.join(lines)

    return ""


def build_experience_latex(experience: list, style: str) -> str:
    """Builds the experience section for a given template style."""
    if not experience:
        return ""

    if style == 'jakes_resume':
        lines = [r'\section{Experience}', r'\resumeSubHeadingListStart']
        for exp in experience:
            title   = safe(exp.get('title', ''))
            company = safe(exp.get('company', ''))
            start   = safe(exp.get('start_date', ''))
            end     = safe(exp.get('end_date', 'Present'))
            date_str = f"{start} -- {end}" if start else end
            lines.append(rf'  \resumeSubheading{{{title}}}{{{date_str}}}{{{company}}}{{}}')
            lines.append(r'  \resumeItemListStart')
            for bullet in exp.get('bullets', []):
                lines.append(rf'    \resumeItem{{{safe(bullet)}}}')
            lines.append(r'  \resumeItemListEnd')
        lines.append(r'\resumeSubHeadingListEnd')
        return '\n'.join(lines)

    elif style in ('simple_professional', 'academic_cv', 'engineering_minimal'):
        section_name = 'Experience' if style != 'academic_cv' else 'Professional Experience'
        lines = [rf'\section{{{section_name}}}']
        for exp in experience:
            title   = safe(exp.get('title', ''))
            company = safe(exp.get('company', ''))
            start   = safe(exp.get('start_date', ''))
            end     = safe(exp.get('end_date', 'Present'))
            date_str = f"{start} -- {end}" if start else end
            lines.append(rf'\textbf{{{company}}} \hfill {safe(date_str)} \\')
            lines.append(rf'\textit{{{title}}}')
            lines.append(r'\begin{itemize}[noitemsep]')
            for bullet in exp.get('bullets', []):
                lines.append(rf'    \item {safe(bullet)}')
            lines.append(r'\end{itemize}')
            lines.append('')
        return '\n'.join(lines)

    elif style == 'moderncv':
        lines = [r'\section{Experience}']
        for exp in experience:
            title   = safe(exp.get('title', ''))
            company = safe(exp.get('company', ''))
            start   = safe(exp.get('start_date', ''))
            end     = safe(exp.get('end_date', 'Present'))
            date_str = f"{start} -- {end}" if start else end
            bullet_items = '\n'.join([rf'    \item {safe(b)}' for b in exp.get('bullets', [])])
            lines.append(rf'\cventry{{{safe(date_str)}}}{{{title}}}{{{company}}}{{}}{{}}{{')
            if bullet_items:
                lines.append(r'\begin{itemize}')
                lines.append(bullet_items)
                lines.append(r'\end{itemize}')
            lines.append(r'}')
        return '\n'.join(lines)

    return ""


def build_projects_latex(projects: list, style: str) -> str:
    """Builds the projects section for a given template style."""
    if not projects:
        return ""

    if style == 'jakes_resume':
        lines = [r'\section{Projects}', r'\resumeSubHeadingListStart']
        for proj in projects:
            name = safe(proj.get('name', 'Project'))
            techs = ', '.join([safe(t) for t in proj.get('technologies', [])])
            tech_str = rf'\emph{{{techs}}}' if techs else ''
            heading = rf'\textbf{{{name}}} $|$ {tech_str}' if tech_str else rf'\textbf{{{name}}}'
            lines.append(rf'  \resumeProjectHeading{{{heading}}}{{}}')
            lines.append(r'  \resumeItemListStart')
            for bullet in proj.get('bullets', []):
                lines.append(rf'    \resumeItem{{{safe(bullet)}}}')
            lines.append(r'  \resumeItemListEnd')
        lines.append(r'\resumeSubHeadingListEnd')
        return '\n'.join(lines)

    elif style in ('simple_professional', 'academic_cv', 'engineering_minimal', 'moderncv'):
        lines = [r'\section{Projects}']
        for proj in projects:
            name  = safe(proj.get('name', 'Project'))
            techs = ', '.join([safe(t) for t in proj.get('technologies', [])])
            lines.append(rf'\textbf{{{name}}}')
            if techs:
                lines.append(rf'\textit{{Technologies: {techs}}}')
            lines.append(r'\begin{itemize}[noitemsep]')
            for bullet in proj.get('bullets', []):
                lines.append(rf'    \item {safe(bullet)}')
            lines.append(r'\end{itemize}')
            lines.append('')
        return '\n'.join(lines)

    return ""


def build_education_latex(education: list, style: str) -> str:
    """Builds the education section for a given template style."""
    if not education:
        return ""

    if style == 'jakes_resume':
        lines = [r'\section{Education}', r'\resumeSubHeadingListStart']
        for edu in education:
            degree  = safe(edu.get('degree', ''))
            field   = safe(edu.get('field', ''))
            inst    = safe(edu.get('institution', ''))
            year    = safe(edu.get('year', ''))
            deg_str = f"{degree} in {field}" if field and field.lower() not in ['unknown', 'not specified', 'n/a'] else degree
            lines.append(rf'  \resumeSubheading{{{inst}}}{{}}')
            lines.append(rf'  {{{deg_str}}}{{{year}}}')
        lines.append(r'\resumeSubHeadingListEnd')
        return '\n'.join(lines)

    elif style == 'moderncv':
        lines = [r'\section{Education}']
        for edu in education:
            degree = safe(edu.get('degree', ''))
            field  = safe(edu.get('field', ''))
            inst   = safe(edu.get('institution', ''))
            year   = safe(edu.get('year', ''))
            deg_str = f"{degree} in {field}" if field and field.lower() not in ['unknown', 'not specified', 'n/a'] else degree
            lines.append(rf'\cventry{{{year}}}{{{deg_str}}}{{{inst}}}{{}}{{}}{{}}')
        return '\n'.join(lines)

    else:  # simple_professional, academic_cv, engineering_minimal
        lines = [r'\section{Education}']
        for edu in education:
            degree = safe(edu.get('degree', ''))
            field  = safe(edu.get('field', ''))
            inst   = safe(edu.get('institution', ''))
            year   = safe(edu.get('year', ''))
            deg_str = f"{degree} in {field}" if field and field.lower() not in ['unknown', 'not specified', 'n/a'] else degree
            lines.append(rf'\textbf{{{deg_str}}} \hfill {year} \\')
            lines.append(rf'\textit{{{inst}}}')
            lines.append('')
        return '\n'.join(lines)


def build_certifications_latex(certifications: list, style: str) -> str:
    """Builds the certifications section for a given template style."""
    if not certifications:
        return ""

    lines = [r'\section{Certifications}']

    if style == 'moderncv':
        for cert in certifications:
            name   = safe(cert.get('name', ''))
            issuer = safe(cert.get('issuer', ''))
            date   = safe(cert.get('date', ''))
            lines.append(rf'\cvitem{{{date}}}{{\textbf{{{name}}} — {issuer}}}')
    else:
        lines.append(r'\begin{itemize}[leftmargin=*, noitemsep]')
        for cert in certifications:
            name   = safe(cert.get('name', ''))
            issuer = safe(cert.get('issuer', ''))
            date   = safe(cert.get('date', ''))
            cert_str = name
            if issuer and issuer.lower() not in ['unknown', 'not specified']:
                cert_str += f' ({issuer})'
            if date and date.lower() not in ['unknown', 'not specified', 'n/a']:
                cert_str += f' — {date}'
            lines.append(rf'    \item {cert_str}')
        lines.append(r'\end{itemize}')

    return '\n'.join(lines)


def build_footer_latex(candidate_name: str, place: str, today_date: str) -> str:
    """Builds the Date / Place / Signature footer block for the bottom of a resume."""
    _name  = safe(candidate_name)
    _place = safe(place) if place else '\\underline{\\hspace{3cm}}'
    _date  = safe(today_date)
    return rf"""
\vspace{{1.5cm}}
\noindent
\begin{{tabular}}{{p{{8cm}} p{{6cm}}}}
\textbf{{Date:}} {_date} & \textbf{{Signature:}} \\
[1cm]
\textbf{{Place:}} {_place} & \textbf{{(Name:}} {_name}\textbf{{)}}
\end{{tabular}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FULL TEMPLATE GENERATORS
# Each function returns a complete .tex string for a given template style.
# ─────────────────────────────────────────────────────────────────────────────

def generate_jakes_resume(data: dict) -> str:
    """Template 1: Jake's Resume — Developer Gold Standard."""
    name       = safe(data['name'])
    email      = safe(data['email'])
    phone      = safe(data['phone'])
    linkedin   = safe(data['linkedin'])
    github     = safe(data['github'])
    summary    = safe(data['summary'])
    _place_val = data.get('place', '')
    _date_val  = data.get('today_date', '')
    footer_tex = build_footer_latex(data['name'], _place_val, _date_val)

    contact_parts = []
    if phone:    contact_parts.append(phone)
    if email:    contact_parts.append(rf'\href{{mailto:{data["email"]}}}{{\underline{{{email}}}}}')
    if linkedin: contact_parts.append(rf'\href{{https://linkedin.com/in/{data["linkedin"]}}}{{\underline{{{linkedin}}}}}')
    if github:   contact_parts.append(rf'\href{{https://github.com/{data["github"]}}}{{\underline{{{github}}}}}')
    contact_line = ' $|$ '.join(contact_parts)

    skills_tex  = build_skills_latex(data['skills_categorized'], 'jakes_resume')
    exp_tex     = build_experience_latex(data['experience'], 'jakes_resume')
    proj_tex    = build_projects_latex(data['projects'], 'jakes_resume')
    edu_tex     = build_education_latex(data['education'], 'jakes_resume')
    cert_tex    = build_certifications_latex(data['certifications'], 'jakes_resume')

    opt_summary = ""
    if summary:
        opt_summary = rf"""
\section{{Professional Summary}}
\small{{{summary}}}
"""

    return rf"""
%-------------------------
% Jake's Resume (ATS-Safe)
% Template: jakes_resume
%-------------------------
\documentclass[letterpaper,11pt]{{article}}
\usepackage{{latexsym}}
\usepackage[empty]{{fullpage}}
\usepackage{{titlesec}}
\usepackage[usenames,dvipsnames]{{color}}
\usepackage{{enumitem}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage[english]{{babel}}
\usepackage{{tabularx}}
\input{{glyphtounicode}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyfoot{{}}
\renewcommand{{\headrulewidth}}{{0pt}}
\renewcommand{{\footrulewidth}}{{0pt}}
\addtolength{{\oddsidemargin}}{{-0.5in}}
\addtolength{{\evensidemargin}}{{-0.5in}}
\addtolength{{\textwidth}}{{1in}}
\addtolength{{\topmargin}}{{-.5in}}
\addtolength{{\textheight}}{{1.0in}}
\urlstyle{{same}}
\raggedbottom\raggedright
\setlength{{\tabcolsep}}{{0in}}
\titleformat{{\section}}{{\vspace{{-4pt}}\scshape\raggedright\large}}{{}}{{0em}}{{}}[\color{{black}}\titlerule \vspace{{-5pt}}]
\pdfgentounicode=1

\newcommand{{\resumeItem}}[1]{{\item\small{{#1 \vspace{{-2pt}}}}}}
\newcommand{{\resumeSubheading}}[4]{{
  \vspace{{-2pt}}\item
    \begin{{tabular*}}{{0.97\textwidth}}[t]{{l@{{\extracolsep{{\fill}}}}r}}
      \textbf{{#1}} & #2 \\
      \textit{{\small#3}} & \textit{{\small #4}} \\
    \end{{tabular*}}\vspace{{-7pt}}}}
\newcommand{{\resumeProjectHeading}}[2]{{
    \item
    \begin{{tabular*}}{{0.97\textwidth}}{{l@{{\extracolsep{{\fill}}}}r}}
      \small#1 & #2 \\
    \end{{tabular*}}\vspace{{-7pt}}}}
\newcommand{{\resumeSubItem}}[1]{{\resumeItem{{#1}}\vspace{{-4pt}}}}
\renewcommand\labelitemii{{$\vcenter{{\hbox{{\tiny$\bullet$}}}}$}}
\newcommand{{\resumeSubHeadingListStart}}{{\begin{{itemize}}[leftmargin=0.15in, label={{}}]}}
\newcommand{{\resumeSubHeadingListEnd}}{{\end{{itemize}}}}
\newcommand{{\resumeItemListStart}}{{\begin{{itemize}}}}
\newcommand{{\resumeItemListEnd}}{{\end{{itemize}}\vspace{{-5pt}}}}

\begin{{document}}

\begin{{center}}
    \textbf{{\Huge \scshape {name}}} \\ \vspace{{1pt}}
    \small {contact_line}
\end{{center}}

{opt_summary}
{edu_tex}
{skills_tex}
{exp_tex}
{proj_tex}
{cert_tex}

{footer_tex}

\end{{document}}
"""


def generate_simple_professional(data: dict) -> str:
    """Template 2: Simple Professional Resume — Maximum ATS Safety."""
    name       = safe(data['name'])
    email      = safe(data['email'])
    phone      = safe(data['phone'])
    linkedin   = safe(data['linkedin'])
    github     = safe(data['github'])
    summary    = safe(data['summary'])
    _place_val = data.get('place', '')
    _date_val  = data.get('today_date', '')
    footer_tex = build_footer_latex(data['name'], _place_val, _date_val)

    contact_parts = []
    if phone:    contact_parts.append(phone)
    if email:    contact_parts.append(rf'\href{{mailto:{data["email"]}}}{{{email}}}')
    if linkedin: contact_parts.append(rf'\href{{https://linkedin.com/in/{data["linkedin"]}}}{{LinkedIn}}')
    if github:   contact_parts.append(rf'\href{{https://github.com/{data["github"]}}}{{GitHub}}')
    contact_line = ' $|$ '.join(contact_parts)

    skills_tex = build_skills_latex(data['skills_categorized'], 'simple_professional')
    exp_tex    = build_experience_latex(data['experience'], 'simple_professional')
    proj_tex   = build_projects_latex(data['projects'], 'simple_professional')
    edu_tex    = build_education_latex(data['education'], 'simple_professional')
    cert_tex   = build_certifications_latex(data['certifications'], 'simple_professional')

    opt_summary = ""
    if summary:
        opt_summary = rf"""
\section{{Professional Summary}}
{summary}
"""

    return rf"""
%-------------------------
% Simple Professional Resume (ATS-Safe)
% Template: simple_professional_resume
%-------------------------
\documentclass[letterpaper,10pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{enumitem}}
\usepackage{{titlesec}}
\usepackage[hidelinks]{{hyperref}}
\titleformat{{\section}}{{\large\bfseries}}{{}}{{0em}}{{}}[\titlerule]
\titlespacing{{\section}}{{0pt}}{{12pt}}{{8pt}}
\setlength{{\parindent}}{{0pt}}

\begin{{document}}

\begin{{center}}
    {{\Huge \textbf{{{name}}}}} \\ \vspace{{5pt}}
    {contact_line}
\end{{center}}

{opt_summary}
{edu_tex}
{skills_tex}
{exp_tex}
{proj_tex}
{cert_tex}

{footer_tex}

\end{{document}}
"""


def generate_academic_cv(data: dict) -> str:
    """Template 3: Classic Academic CV — Linear & Detailed."""
    name     = safe(data['name'])
    email    = safe(data['email'])
    github   = safe(data['github'])
    summary  = safe(data['summary'])

    skills_tex = build_skills_latex(data['skills_categorized'], 'academic_cv')
    exp_tex    = build_experience_latex(data['experience'], 'academic_cv')
    proj_tex   = build_projects_latex(data['projects'], 'academic_cv')
    edu_tex    = build_education_latex(data['education'], 'academic_cv')
    cert_tex   = build_certifications_latex(data['certifications'], 'academic_cv')

    opt_summary = ""
    if summary:
        opt_summary = rf"""
\section{{Profile Summary}}
{summary}
"""

    github_line = rf' $|$ Website: \href{{https://github.com/{data["github"]}}}{{github.com/{data["github"]}}}' if github else ""

    return rf"""
%-------------------------
% Classic Academic CV
% Template: academic_cv_article
%-------------------------
\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{enumitem}}
\usepackage{{titlesec}}
\usepackage[hidelinks]{{hyperref}}
\titleformat{{\section}}{{\large\bfseries\uppercase}}{{}}{{0em}}{{}}[\titlerule]
\titlespacing{{\section}}{{0pt}}{{15pt}}{{10pt}}
\setlength{{\parindent}}{{0pt}}

\begin{{document}}

\begin{{center}}
    {{\LARGE \textbf{{{name}}}}} \\ \vspace{{5pt}}
    Email: \href{{mailto:{data["email"]}}}{{{email}}}{github_line}
\end{{center}}

{opt_summary}
{edu_tex}
{skills_tex}
{exp_tex}
{proj_tex}
{cert_tex}

{footer_tex}

\end{{document}}
"""


def generate_moderncv(data: dict) -> str:
    """Template 4: ModernCV (Cleaned) — Single-column, no sidebar."""
    name_parts = data['name'].split()
    first_name = safe(name_parts[0]) if name_parts else 'First'
    last_name  = safe(' '.join(name_parts[1:])) if len(name_parts) > 1 else 'Last'

    email      = data['email']
    phone      = data['phone']
    linkedin   = data['linkedin']
    github     = data['github']
    summary    = safe(data['summary'])
    _place_val = data.get('place', '')
    _date_val  = data.get('today_date', '')
    footer_tex = build_footer_latex(data['name'], _place_val, _date_val)

    skills_tex = build_skills_latex(data['skills_categorized'], 'moderncv')
    exp_tex    = build_experience_latex(data['experience'], 'moderncv')
    proj_tex   = build_projects_latex(data['projects'], 'moderncv')
    edu_tex    = build_education_latex(data['education'], 'moderncv')
    cert_tex   = build_certifications_latex(data['certifications'], 'moderncv')

    phone_line    = rf'\phone[mobile]{{{safe(phone)}}}' if phone else ''
    email_line    = rf'\email{{{safe(email)}}}' if email else ''
    linkedin_line = rf'\social[linkedin]{{{safe(linkedin)}}}' if linkedin else ''
    github_line   = rf'\social[github]{{{safe(github)}}}' if github else ''

    opt_summary = ""
    if summary:
        opt_summary = rf'\section{{Professional Summary}}{summary}'

    return rf"""
%-------------------------
% ModernCV (Cleaned, Banking Style)
% Template: moderncv_clean
%-------------------------
\documentclass[11pt,a4paper,sans]{{moderncv}}
\moderncvstyle{{banking}}
\moderncvcolor{{blue}}
\usepackage[utf8]{{inputenc}}
\usepackage[scale=0.75]{{geometry}}

\name{{{first_name}}}{{{last_name}}}
{phone_line}
{email_line}
{linkedin_line}
{github_line}

\begin{{document}}
\makeheader

{opt_summary}
{edu_tex}
{skills_tex}
{exp_tex}
{proj_tex}
{cert_tex}

{footer_tex}

\end{{document}}
"""


def generate_engineering_minimal(data: dict) -> str:
    """Template 5: Minimal Engineering Resume — Project-heavy layout."""
    name       = safe(data['name'])
    email      = safe(data['email'])
    phone      = safe(data['phone'])
    linkedin   = safe(data['linkedin'])
    github     = safe(data['github'])
    summary    = safe(data['summary'])
    _place_val = data.get('place', '')
    _date_val  = data.get('today_date', '')
    footer_tex = build_footer_latex(data['name'], _place_val, _date_val)

    contact_parts = []
    if phone:    contact_parts.append(phone)
    if email:    contact_parts.append(email)
    if github:   contact_parts.append(rf'\href{{https://github.com/{data["github"]}}}{{github.com/{data["github"]}}}')
    if linkedin: contact_parts.append(rf'\href{{https://linkedin.com/in/{data["linkedin"]}}}{{linkedin.com/in/{data["linkedin"]}}}')
    contact_line = ' $|$ '.join(contact_parts)

    # Engineering Minimal puts Skills FIRST, then Experience, then Projects
    skills_tex = build_skills_latex(data['skills_categorized'], 'engineering_minimal')
    exp_tex    = build_experience_latex(data['experience'], 'engineering_minimal')
    proj_tex   = build_projects_latex(data['projects'], 'engineering_minimal')
    edu_tex    = build_education_latex(data['education'], 'engineering_minimal')
    cert_tex   = build_certifications_latex(data['certifications'], 'engineering_minimal')

    opt_summary = ""
    if summary:
        opt_summary = rf"""
\section{{Objective}}
{summary}
"""

    return rf"""
%-------------------------
% Minimal Engineering Resume
% Template: engineering_minimal
%-------------------------
\documentclass[10pt,letterpaper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{titlesec}}
\usepackage{{enumitem}}
\usepackage[hidelinks]{{hyperref}}
\titleformat{{\section}}{{\large\scshape\raggedright}}{{}}{{0em}}{{}}[\titlerule]
\titlespacing{{\section}}{{0pt}}{{10pt}}{{5pt}}
\setlength{{\parindent}}{{0pt}}

\begin{{document}}

\begin{{center}}
    {{\Huge {name}}} \\ \vspace{{5pt}}
    {contact_line}
\end{{center}}

{opt_summary}
{edu_tex}
{skills_tex}
{exp_tex}
{proj_tex}
{cert_tex}

{footer_tex}

\end{{document}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: AI VERIFICATION (Groq LLaMA-3)
# Sends generated LaTeX to LLaMA-3 for syntax checking and auto-fix.
# ─────────────────────────────────────────────────────────────────────────────

def verify_latex_with_llm(latex_code: str, groq_client) -> tuple[str, str]:
    """
    Sends the generated LaTeX code to LLaMA-3 for syntax verification.

    Returns:
        (verified_code, status_message)
        - If clean: returns original code + "CLEAN"
        - If fixed:  returns fixed code + "FIXED"
        - If error:  returns original code + "SKIP"
    """
    verification_prompt = f"""You are a LaTeX expert and compiler. 
Below is a generated resume LaTeX document.

YOUR TASK:
1. Scan for ANY syntax errors: missing \\end{{}} tags, unbalanced braces {{}}, unescaped special characters (& % $ # _ {{ }} ~ ^), or invalid commands.
2. If no errors found → Return EXACTLY the word: FIX_NOT_REQUIRED
3. If errors found → Return the COMPLETE corrected LaTeX code. Nothing else.

IMPORTANT:
- Do NOT change the resume content or structure.
- Do NOT add comments or explanations.
- Only fix actual LaTeX syntax issues.

LATEX CODE TO CHECK:
```latex
{latex_code}
```"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a LaTeX syntax checker. Return 'FIX_NOT_REQUIRED' if the code is clean, otherwise return only the fully corrected LaTeX code."
                },
                {"role": "user", "content": verification_prompt}
            ],
            temperature=0.0,   # Deterministic for code verification
            max_tokens=4000
        )

        llm_response = response.choices[0].message.content.strip()

        if "FIX_NOT_REQUIRED" in llm_response:
            return latex_code, "CLEAN"
        else:
            # LLM returned corrected code - extract the LaTeX block
            # Sometimes LLM wraps in ```latex ... ```
            code_match = re.search(r'```(?:latex)?\s*(.*?)```', llm_response, re.DOTALL)
            if code_match:
                fixed_code = code_match.group(1).strip()
            else:
                fixed_code = llm_response.strip()
            return fixed_code, "FIXED"

    except Exception as e:
        # If API call fails, return original code and skip verification
        return latex_code, f"SKIP (API error: {str(e)[:50]})"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: TEMPLATE REGISTRY
# Maps template_id → (display_name, generator_function, description)
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_REGISTRY = {
    "jakes_resume": {
        "display_name": "Jake's Resume",
        "type": "Developer Resume",
        "ats_safety": "⭐⭐⭐⭐⭐",
        "best_for": "Balanced — Strong look + High ATS safety",
        "generator": generate_jakes_resume,
        "image": "assets/template_1.jpg"
    },
    "simple_professional": {
        "display_name": "Simple Professional",
        "type": "Minimal Resume",
        "ats_safety": "⭐⭐⭐⭐⭐",
        "best_for": "Safest ATS — Maximum parser compatibility",
        "generator": generate_simple_professional,
        "image": "assets/template_2.jpg"
    },
    "academic_cv": {
        "display_name": "Classic Academic CV",
        "type": "Academic / Data Science",
        "ats_safety": "⭐⭐⭐⭐",
        "best_for": "Research or Data Science profiles",
        "generator": generate_academic_cv,
        "image": "assets/template_3.jpg"
    },
    "moderncv": {
        "display_name": "ModernCV (Cleaned)",
        "type": "Professional Resume",
        "ats_safety": "⭐⭐⭐",
        "best_for": "Slight visual enhancement with ATS safety",
        "generator": generate_moderncv,
        "image": "assets/template_4.jpg"
    },
    "engineering_minimal": {
        "display_name": "Engineering Minimal",
        "type": "Software / AI Resume",
        "ats_safety": "⭐⭐⭐",
        "best_for": "Project-heavy candidates in Software / AI",
        "generator": generate_engineering_minimal,
        "image": "assets/template_5.jpg"
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: STREAMLIT UI COMPONENT
# Call this function from within the main app.py results section.
# ─────────────────────────────────────────────────────────────────────────────

def show_latex_phase(optimized_resume: dict, resume_struct: dict, groq_client):
    """
    Full Streamlit UI for Phase 10: LaTeX Resume Generation.

    Shows:
      - 5 template previews with images, name, ATS rating
      - "Generate LaTeX" button
      - AI verification step
      - Download button for the final .tex file

    Usage in app.py:
        from latex_module import show_latex_phase
        show_latex_phase(res['optimized_content'], res['resume_struct'], groq_client)
    """

    st.markdown("---")
    st.markdown("### 📄 Phase 10: LaTeX Resume Generator")
    st.markdown("Select a template style. Your resume data will be injected automatically and verified by AI.")

    # ── Display template cards (5 columns) ──
    template_ids = list(TEMPLATE_REGISTRY.keys())
    cols = st.columns(len(template_ids))

    selected_template = st.session_state.get('selected_latex_template', None)

    for i, (tid, info) in enumerate(TEMPLATE_REGISTRY.items()):
        with cols[i]:
            # Show preview image if it exists
            img_path = os.path.join(os.path.dirname(__file__), info['image'])
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.markdown(f"*Preview*")

            st.markdown(f"**{info['display_name']}**")
            st.caption(f"{info['type']}")
            st.caption(f"ATS: {info['ats_safety']}")
            st.caption(f"*{info['best_for']}*")

            # Selection button
            btn_label = "✅ Selected" if selected_template == tid else "Select"
            if st.button(btn_label, key=f"latex_select_{tid}", use_container_width=True):
                st.session_state['selected_latex_template'] = tid
                st.rerun()

    st.markdown("---")

    # ── Generate section ──
    if selected_template and selected_template in TEMPLATE_REGISTRY:
        info = TEMPLATE_REGISTRY[selected_template]
        st.success(f"✅ Template Selected: **{info['display_name']}** ({info['type']})")

        if st.button(" Generate LaTeX File", type="primary", use_container_width=True):

            with st.spinner(" Extracting resume data..."):
                data = extract_resume_data(optimized_resume, resume_struct)

            with st.spinner(f" Generating {info['display_name']} template..."):
                generator_fn = info['generator']
                raw_latex = generator_fn(data)

            with st.spinner("🤖 AI Verification with LLaMA-3 (checking for LaTeX syntax errors)..."):
                verified_latex, status = verify_latex_with_llm(raw_latex, groq_client)

            # Status feedback
            if status == "CLEAN":
                st.success("✅ AI Verified: No LaTeX errors found!")
            elif status == "FIXED":
                st.warning("🔧 AI Auto-Fixed: LLaMA-3 corrected minor LaTeX syntax issues.")
            else:
                st.info(f"ℹ️ Verification skipped: {status}. File generated as-is.")

            # Preview the code
            with st.expander(" Preview Generated LaTeX Code"):
                st.code(verified_latex, language='latex')

            # Download button
            candidate_name = data['name'].replace(' ', '_') if data['name'] else 'Resume'
            filename = f"{candidate_name}_{selected_template}_ATS.tex"

            st.download_button(
                label="⬇️ Download .tex File (Overleaf Ready)",
                data=verified_latex.encode('utf-8'),
                file_name=filename,
                mime="text/plain",
                use_container_width=True
            )

            st.info(
                "💡 **How to use:** Upload this `.tex` file to [Overleaf](https://www.overleaf.com) "
                "and click **Recompile** to get your professional PDF resume."
            )

    else:
        st.info("👆 Select a template above to generate your LaTeX resume.")