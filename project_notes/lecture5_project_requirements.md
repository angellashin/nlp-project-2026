# NLP Final Project Requirements and Working Notes

Source: `lecture 5.pdf`, pages 2-38, plus a light pass over `Good Final Report Samples 2024/` and `Good Final Report Samples 2025/`.

Current working date: 2026-04-28.

## 1. Core Goal

The final project should be a small but real research project involving human language and neural networks. The expected output is not just a demo, but a research-style result with a clear question, method, experiment, and written report.

The project can start from either direction:

- **Problem-first ("nails")**: choose a domain/task involving language, then find a good or better way to solve it.
- **Method-first ("hammers")**: choose a technical method/model you care about, then extend, improve, analyze, or apply it in a meaningful way.

The professor emphasizes being realistic. Colab-scale or 1-GPU experiments are acceptable. The project does not need to beat large labs, but it should ask an interesting question and support the answer with careful experiments.

## 2. Team and Scope Expectations

- Team size can be 1-4 people.
- Larger teams are expected to produce larger results.
- Larger result can mean exploring more models, tasks, datasets, ablations, or analyses.
- Any programming language or framework is allowed.
- PyTorch is expected/normal, but not mandatory.
- Transformers/Hugging Face are treated as the default practical toolkit for 2026 NLP projects.
- Compute should be scoped conservatively. Training huge models from scratch is not realistic.

## 3. Important Dates From Lecture

- Proposal presentations: 2026-04-03 and 2026-04-10.
- Final report due: 2026-05-29.
- Selected teams announced: 2026-06-05.
- Selected team presentations: 2026-06-12 and 2026-06-19.

Because today is 2026-04-28, the proposal presentation dates have already passed, but the report deadline and possible final presentations are still ahead.

## 4. Proposal Expectations

The proposal should be concise and concrete. A good proposal usually does the following:

1. Identify a relevant key research paper.
2. Summarize that paper briefly, especially the key idea worth keeping.
3. Explain what we will work on and how we will innovate.
4. Give a sensible halfway milestone.
5. Describe the project plan, related literature, models to use/explore, data source, and evaluation method.

The professor's warning is that weak proposals often fail in practical details, not in ambition. The idea must be sensible, but the plan also needs data, baselines, evaluation, and ablations.

## 5. How To Read Papers For This Project

When choosing or adapting a paper, we should ask:

- What exactly is novel?
- Is the core idea general and reusable?
- Are there flaws, hidden assumptions, or interesting implementation details?
- How does it relate to similar papers?
- Does it suggest good follow-up experiments or variants?

The project should probably preserve one key insight from a paper, then modify the setting, data, model, objective, efficiency constraint, or analysis.

## 6. Acceptable Project Types

The lecture lists several common project shapes:

- **Application/task project**: choose a language-related task and solve or improve it, often with existing models.
- **Architecture implementation project**: implement a complex neural architecture and show performance on data.
- **Model variant project**: propose a new or modified neural model and empirically test it.
- **Analysis project**: analyze model behavior, linguistic knowledge, robustness, errors, bias, explainability, etc.
- **Theoretical project**: show a non-trivial property of a model, data, or representation. This is rare.

For the first three types, experiments with numerical results and ablation studies are expected.

## 7. Strong Topic Directions Mentioned

The lecture suggests that modern NLP projects should usually work with or around pretrained models rather than building everything from scratch.

Promising directions:

- Robustness to domain shift.
- Robustness evaluation in general.
- What large pretrained models have learned.
- Transfer learning or low-data adaptation.
- Bias, trustworthiness, and explainability.
- Low-resource languages or low-resource tasks.
- Long-tail performance.
- LLM topics: agents, sycophancy, hallucination, factuality, reasoning.
- Scaling models down: pruning, quantization, efficient QA, small performant models.
- Advanced functionality on smaller problems: compositionality, systematic generalization, fast learning, meta-learning.

The professor explicitly says big-model training is not realistic for a course project. Small, efficient, well-scoped model work may be a strong direction.

## 8. Using GPT, ChatGPT, or APIs

Using GPT-3/ChatGPT-style systems is allowed, but the evaluation is based on what we do, not on impressive zero-shot outputs.

Good ways to use them:

- Analysis projects.
- Prompt learning / prompt engineering as a systematic experiment.
- Chain-of-thought or reasoning studies.
- Hallucination, factuality, trustworthiness, or evaluation work.

Risks:

- API costs are not funded.
- "Look, it works zero-shot" is not enough.
- The project needs a research contribution, evaluation, or careful analysis.

## 9. Data Expectations

Possible data sources:

- Existing curated research datasets.
- Kaggle, bake-offs, shared tasks, or benchmark datasets.
- Data from a student's own major/domain.
- Self-collected data.
- Small manually annotated data.
- Websites with natural labels such as likes, stars, ratings, responses, or user interactions.
- Existing data from a company/research project, if samples can be submitted or described.
- Unsupervised data.

Professor's preference:

- Collecting data can be valuable, but it can consume too much time.
- Existing curated datasets give a faster start and usually provide prior work and baselines.
- If collecting data, scope it tightly.

Dataset resources mentioned:

- Hugging Face Datasets.
- WMT/shared tasks for machine translation.
- Kaggle.
- Papers' dataset sections.
- GLUE tasks.
- Stanford Sentiment Treebank.
- Facebook bAbI-related datasets.
- Dataset lists such as NLP dataset repositories.

## 10. Evaluation and Splits

Valid evaluation is a major requirement.

Key rules:

- Train/dev/test must be distinct.
- Do not test on training material.
- Use a tuning/validation/dev set for progress and hyperparameters.
- Use the final test set very few times, ideally once.
- If no dev set exists, split training data.
- With small data, cross-validation can help.
- Repeatedly checking the same evaluation set can overfit the experiment to that set.
- A second dev set can help when many iterations are expected.

The final report should make clear:

- Which split was used.
- What metric measures success.
- What baselines are compared.
- What ablations or variants isolate the contribution.

## 11. Experimental Strategy

The lecture's debugging strategy is very practical:

- Start simple.
- Get a very simple model or baseline working first.
- Add complexity one component at a time.
- Initially run on a tiny dataset, around 4-8 examples.
- Make sure the model can overfit tiny data to 100%.
- If it cannot, the model/data/training loop is probably broken or underpowered.
- Then scale to the full dataset.
- Check whether training performance gets close to 100% after optimization.
- Regularize to improve dev performance.
- Learning rate, initialization, hyperparameters, and dropout matter.

Important habit:

- Inspect the data.
- Collect summary statistics.
- Inspect model outputs.
- Do error analysis.

## 12. Final Report Expectations

Report quality is very important to the grade.

Format:

- LaTeX template will be provided.
- Other editors such as MS Word are allowed.
- Length: 1-8 pages.

Expected report structure from the lecture example:

- Abstract.
- Introduction.
- Prior related work.
- Data.
- Model.
- Experiments.
- Results.
- Analysis and conclusion.

Practical implication:

The report should read like a compact research paper. It should clearly state the problem, what was tried, why it was tried, what data/evaluation was used, what the results show, and what limitations remain.

## 13. Example Projects Shown In Lecture

Pages 10-16 show selected examples from previous final projects. The examples suggest the professor values projects that are ambitious but still scoped around concrete model changes, efficiency, evaluation, or domain adaptation.

Examples shown:

- **Dual-CoCoOp: Dually-informed Conditional Context Optimization for Vision-Language Models**
  - Vision-language model adaptation.
  - Architecture-level method with frozen and trainable components.

- **HIES: Joint Importance-Entropy Pruning for Transformer Heads**
  - Efficiency/compression project.
  - Compares pruning behavior on tasks such as SST-2 and CoLA.

- **HanCLIP / uCLIP: Korean or multilingual vision-language extension**
  - Multilingual/low-resource vision-language adaptation.
  - Slide notes this line was published in AAAI 2026 as uCLIP.

- **KoLa: Korean speech to LaTeX extraction using small LM modules**
  - Applied pipeline project.
  - Uses modular ASR, correction, and LaTeX translation.

- **LaCon: Layer-Contrastive Decoding for mitigating hallucinations in vision-language models**
  - Hallucination/factuality project.
  - Modifies decoding behavior rather than training a giant model.

- **PIN Me If You Can: Robust lip-only digit recognition using hybrid ViT-RNN architecture**
  - Multimodal/speech-adjacent recognition project.
  - Uses hybrid architecture and robustness framing.

- **ConRaGen: Domain-specific contrastive radiology-report generation**
  - Domain-specific generation.
  - Combines medical text/report generation with contrastive or retrieval-like signals.

Common pattern across examples:

- Clear task/domain.
- Named model or method contribution.
- Diagram or system overview.
- Quantitative results.
- Comparison against baselines or variants.
- A concrete reason the project is not just "use a pretrained model."

## 14. Sample Report Folder Observations

The workspace contains:

- `Good Final Report Samples 2024/`
- `Good Final Report Samples 2025/`

There is no folder literally named `samples`; the available sample material appears to be these two folders.

Observed sample report titles include:

- PicassoGen: Picture Assembler System using Stepwise Object Arrangement.
- Solving Korea Dialect Translation Problem Under Data Scarcity.
- Rationalizing Common Intuition: Writing Daangn Market Posts that "Click".
- Mental health screening using fine-tuned SBERT.
- Enhancing sarcasm detection with pretrained LMs and multi-task learning.
- PLAR-3D: Point-Language Aligned Rewards for Multi-View Text-to-3D.
- KISS: Keyword-Informed Sharding Strategy for Machine Unlearning.
- LAPD: LLM Alignment with Persona Dynamics.
- KoDistillE5: Korean-English Cross-Lingual Information Retrieval.
- DPO-based Korean language detoxification with context-appropriate emojis.
- Dual-CoCoOp.
- HIES.
- HanCLIP.
- KoLa.
- LaCon.
- PIN Me If You Can.
- ConRaGen.

The samples reinforce that strong projects usually have a memorable title, a compact research-paper format, and a focused contribution.

## 15. Working Checklist For Our Project

Before committing to a topic, verify:

- Is the task clearly related to human language and neural networks?
- Is there a key paper or technical anchor?
- Is the dataset available or realistically collectible?
- Is the compute requirement realistic?
- Is there a simple baseline?
- Is there at least one stronger baseline or prior method to compare against?
- Are there meaningful ablations?
- Is success measurable with clear metrics?
- Can we produce interesting error analysis?
- Can the whole project fit before the 2026-05-29 report deadline?

For implementation:

- Build the simplest baseline first.
- Run tiny-data overfit tests.
- Keep train/dev/test split clean.
- Log metrics and experiment settings.
- Save representative model outputs.
- Plan ablations early.
- Reserve time for report writing and figures.

For the final report:

- Tell a clear research story.
- Include enough related work to position the contribution.
- Describe data and evaluation precisely.
- Include baselines and ablations.
- Report numbers honestly.
- Add qualitative examples/error analysis.
- State limitations and future work.

## 16. Practical Project Shape To Prefer

Given the lecture guidance, a strong course-sized project will likely look like one of these:

- Fine-tune or adapt a pretrained model to a specific language/domain/task, with careful baselines and error analysis.
- Evaluate robustness, bias, hallucination, factuality, or reasoning behavior of an existing model, then test a mitigation.
- Build a small/efficient model variant through pruning, quantization, distillation, LoRA, adapters, or decoding changes.
- Create or annotate a small focused dataset, then use it to reveal a model weakness or improve a targeted task.
- Apply an NLP method to a domain-specific dataset from another field, with a clear evaluation target.

Avoid project shapes that are mostly:

- Training a large model from scratch.
- Showing impressive LLM outputs without systematic evaluation.
- Building a demo without quantitative results.
- Collecting data so broadly that there is no time left for modeling and analysis.
- Comparing models without explaining why the comparison answers a research question.

