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
│   ├── pre-req.md            # Pre-requisite topics
│   ├── study/
│   │   └── README.md
│   └── questions/
│       └── README.md
├── NLPA/                     # NLP Applications
│   ├── README.md
│   ├── pre-req.md            # Pre-requisite topics
│   ├── study/
│   │   └── README.md
│   └── questions/
│       └── README.md
├── CAI/                      # Conversational AI
│   ├── README.md
│   ├── pre-req.md            # Pre-requisite topics
│   ├── study/
│   │   └── README.md
│   └── questions/
│       └── README.md
└── SMA/                      # Social Media Analytics
    ├── README.md
    ├── pre-req.md            # Pre-requisite topics
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

### Pre-requisites (`pre-req.md`)
- Every course folder **must** contain a `pre-req.md` file
- Lists topics assumed to be learnt before starting the course
- Organized by category (e.g., Mathematics, Machine Learning, Deep Learning, NLP, Programming)
- Each category uses a `##` heading; individual topics are bullet points
- Update when new prerequisite knowledge is identified

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

## Study Material Linking Rules

When adding or updating "Topics to Know" sections in question files:

1. **Link to specific sub-headings** using markdown heading anchors, NOT the top-level file
   - ✅ `📖 [5.4.2 NICE / RealNVP](../study/05-normalizing-flow-models.md#542-nice--realnvp)`
   - ❌ `📖 [Normalizing Flow Models](../study/05-normalizing-flow-models.md)`

2. **Link multiple sub-topics** if a question topic spans several sections, separated by ` · `
   - Example: `📖 [6.3 ELBO](…#63-variational--evidence-lower-bound-elbo) · [6.4 KL Divergence](…#kl-divergence-gaussian-case-closed-form)`

3. **Create missing study content** if a question references a topic not yet covered in the study materials — add sub-sections with formulas, explanations, and comparison tables

4. **Use the 📖 emoji** before the first link for visual consistency

5. **Use relative paths** from questions to study: `../study/{filename}.md#anchor`

6. **Anchor format**: lowercase, hyphens replace spaces, strip special chars (e.g., `### 5.4.2 NICE / RealNVP` → `#542-nice--realnvp`)
