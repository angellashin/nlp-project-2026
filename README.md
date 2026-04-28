# NLP Project 2026

Working project:

**What Context Errors Hurt Small LMs Most?**  
A Controlled Study on RumourEval Stance Detection

This repository is for a Korea University NLP final project studying how small language models respond to useful, irrelevant, conflicting, and mixed conversational context in RumourEval 2019 Task A stance detection.

## Current Documents

- [Project proposal](docs/project_proposal.md): source-of-truth framing, dataset scope, context conditions, model plan, evaluation, risks, and roadmap.
- [Lecture 5 requirements notes](project_notes/lecture5_project_requirements.md): course project expectations summarized from the lecture slides.

## Scope

- Main task: RumourEval 2019 Task A stance classification.
- Labels: `support`, `deny`, `query`, `comment`.
- Main analysis unit: labeled reply nodes, with source posts used as context.
- Main models: small language models around 0.5B and 1.5B parameters.

## Repository Policy

Course-provided PDFs, sample reports, raw datasets, model checkpoints, and local runtime state are intentionally ignored by Git. Keep shared work in `docs/`, `src/`, `configs/`, and lightweight `results/` artifacts.
