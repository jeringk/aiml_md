# Project Memory — AIML MTech Sem3 Study Materials

## Purpose
This repository stores **markdown-only** study materials for the AIML MTech program at BITS Pilani, Semester 3. All content is designed for printout and offline study.

## Semester
- **Program:** MTech in Artificial Intelligence & Machine Learning (AIML)
- **Institution:** BITS Pilani (WILP)
- **Semester:** 3

## Repository Structure

```
MarkDown/
├── README.md                 # General info, exam dates, overview
├── QUESTION_TEMPLATE.md      # Reference template for question formatting
├── ADL/                      # Advanced Deep Learning
│   ├── README.md
│   ├── study/
│   │   └── README.md
│   └── questions/
│       └── README.md
├── NLPA/                     # NLP Applications
│   ├── README.md
│   ├── study/
│   │   └── README.md
│   └── questions/
│       └── README.md
├── CAI/                      # Conversational AI
│   ├── README.md
│   ├── study/
│   │   └── README.md
│   └── questions/
│       └── README.md
└── SMA/                      # Social Media Analytics
    ├── README.md
    ├── study/
    │   └── README.md
    └── questions/
        └── README.md
```

## Courses

| Code | Full Name                  |
|------|----------------------------|
| ADL  | Advanced Deep Learning     |
| NLPA | NLP Applications           |
| CAI  | Conversational AI          |
| SMA  | Social Media Analytics     |

## Content Conventions

### Study Materials (`study/`)
- Topic-wise markdown files
- Clear headings and sub-headings
- All formulas rendered in LaTeX math syntax

### Questions (`questions/`)
- Contains **past question papers** and **generated questions**
- Every question is **numbered sequentially** (Q1, Q2, Q3…)
- Each question spans **3 pages** (separated by page breaks):
  1. **Question** — The question text with marks and source
  2. **Topics to Know** — Key topics to study to answer the question
  3. **Solution** — Step-by-step solution

### Formatting Rules
- **Inline math:** `$...$`
- **Display math:** `$$...$$`
- **Page breaks:** `<div style="page-break-after: always;"></div>`
- **Horizontal rules:** `---` to visually separate sections
- All content must be **print-ready**

## Key Decisions
- Markdown-only repo (no code, no notebooks)
- Page breaks via HTML `<div>` for cross-renderer compatibility
- LaTeX math syntax for all formulas
- Sequential question numbering per file

## Git Workflow
- **Commit after every change** — Every file addition, edit, or deletion must be followed by a `git commit`
- **Use conventional commit messages:** `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, etc.
- **Commit message format:** Brief summary (≤50 chars), then detailed description if needed
- **Never leave uncommitted changes** — The working tree should always be clean after each task
