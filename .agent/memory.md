# AGENTS.md

## Scope
This repository contains markdown-only, print-ready study materials for BITS Pilani WILP AIML Semester 3.

## Program Context
- Program: MTech in Artificial Intelligence & Machine Learning (AIML)
- Institution: BITS Pilani (WILP)
- Semester: 3
- Courses: ADL, NLPA, CAI, SMA

## Repository Expectations
- Keep content markdown-only (no notebooks or code artifacts).
- Preserve the existing course layout:
  - `<COURSE>/index.md`
  - `<COURSE>/pre-req.md`
  - `<COURSE>/study/`
  - `<COURSE>/questions/`
- Ensure all materials remain suitable for offline reading/printing.

## Compatibility Policy
- Whatever changes you make, ensure they are compatible with both GitHub Pages and Obsidian.
- Use standard Markdown that renders in both engines; avoid platform-specific syntax unless explicitly requested.
- Escape Markdown table delimiter characters (`|`) inside normal text/math expressions (for example: `P(x\|y)`, `\|d\|`) to prevent unintended table rendering.

## Formatting Standards
- Inline math: `$...$`
- Display math: `$$...$$`
- Parameter definitions under formulas: use multi-line `where:` blocks with one symbol-definition per bullet (avoid long single-line parameter lists).
- Expand variables in formulas explicitly: for every key formula, define each variable/symbol used (including indices, distributions, and parameters) immediately below the formula.
- Page breaks: `<div style="page-break-after: always;"></div>`
- Visual separators: `---`
- Use clear heading hierarchy and readable sectioning.

## Prerequisite Files (`pre-req.md`)
- Every course must include `pre-req.md`.
- Organize prerequisites by category using `##` headings.
- Use bullet lists for topics.
- Update when new prerequisite knowledge appears.

## Study Content (`study/`)
- Maintain topic-wise markdown files.
- Include formulas in LaTeX syntax.
- Add diagrams where useful; prefer SVG files in `<COURSE>/images/`.

## Question Content (`questions/`)
- Include past and generated questions.
- Number questions sequentially within each file: `Q1`, `Q2`, `Q3`, ...
- Keep each question in a 3-page structure:
  1. Question (with marks/source)
  2. Topics to Know
  3. Solution (step-by-step)
- When transcribing from a question paper/image, copy the **Question page text verbatim**.
- Do **not** summarize, trim, paraphrase, simplify, or rewrite any part of the original question text.
- Preserve original wording, ordering, punctuation, labels, marks split, and listed data exactly as provided.
- Apply the verbatim transcription rule before adding "Topics to Know" and "Solution" sections.

## SVG Guidelines
- Store SVG files under `<COURSE>/images/`.
- Embed with markdown image syntax.
- Create grayscale-only visuals optimized for black-and-white printing.
- Ensure strong contrast using black, white, and gray tones only (no color accents).
- Prefer clear educational visuals (~800px wide, readable labels, minimal decoration).
- Use descriptive file names like `topic-subtopic.svg`.

## Linking and Obsidian Rules
- Obsidian is the primary viewer.
- Heading links must use exact-text Obsidian wikilinks:
  - `[[#Exact Heading Text|Display Text]]`
- Cross-file heading links:
  - `[[filename#Heading|Display Text]]`
- Use exact, case-sensitive heading text in wikilinks.
- Standard markdown links are acceptable for external links and file-level links.

## "Topics to Know" Linking Rules
When adding/updating links in question files:
1. Link to specific study sub-headings, not only top-level files.
2. If needed, include multiple sub-topic links separated by ` · `.
3. If referenced topic content is missing, add it to study materials.
4. Prefix the first study link with `📖`.
5. Use relative Obsidian-compatible paths (omit `.md`).

## Git Workflow Policy
- Use a separate git worktree per conversation/task.
- Commit after every change.
- Use conventional commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
- Keep commit summaries concise (<=50 characters), with details in body when needed.
- Avoid leaving uncommitted changes.

## Agent-Memory Sync Policy
- Treat `/Users/jingo/Library/CloudStorage/GoogleDrive-jeringeok@gmail.com/My Drive/Personal/Learning/AIML/AIMLBits/Sem3/MarkDown/AGENTS.md` and `/Users/jingo/Library/CloudStorage/GoogleDrive-jeringeok@gmail.com/My Drive/Personal/Learning/AIML/AIMLBits/Sem3/MarkDown/.agent/memory.md` as synchronized policy files.
- Any rule added, removed, or changed in one file must be reflected in the other in the same update session.
- Before finalizing a task, verify both files are aligned on shared instructions and do not conflict.
