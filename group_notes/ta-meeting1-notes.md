# TA Meeting Notes — 24 April 2026

**Attendees:** Will, Tijje, Haukur, Efe | **TA:** Sebastian Pusch (s.pusch@student.rug.nl)  
**Duration:** 10:30–11:10 (40 min)

For Haukur's transcription of the meeting, see [here](./ta-meeting1-24-04-26.md).

---

## Project Feedback

**Verdict: Approved & within scope**
- TA endorsed the project as justified and useful
- Fits course scope well, particularly due to novelty aspect
- Good balance of challenge and feasibility

---

## Key Decisions & Guidance

### Compression Strategy

- **Next step:** Research good starting approaches; send literature to TA for feasibility check
- **Bottleneck:** Focus compression on the model itself, not embeddings (embedding compression is secondary)
- **Scope:** Position on-edge model (with compression, quantization, pruning) as a deployment extension, not core requirement

### Model Training & Evaluation

- Training process & loss function approach is sound
- **Evaluation:** Match original paper methodology, but feel free to add additional evaluation methods
- **Baseline:** FFT + SVM baseline is acceptable; optional to replicate original paper's baseline for comparison

### Data Split & Class Balancing

- Question about 80/20 person-wise train/test split wasn't directly answered, but raise this for clarification

### Presentation

- **Recommendation:** Use a video demo instead of live "on stage" phone demo (safer, less risk)
- Live phone demo possible but risky
- Presentation quality is not graded; focus on clear communication of results

---

## Timeline & Scope Management

- **Current timeline:** Ambitious but feasible if training estimates are realistic
- **Flexibility:** Can adjust deadlines after week 3–4 if needed
- **Key advice:** Identify bottlenecks early (e.g., compression challenges) and weight effort accordingly, especially for first-time implementations

---

## Repository & DevOps

### Minimize Extra Infrastructure

- **CI/CD recommendation:** Keep simple (linters only) if pursuing; complex pipelines risk wasted effort with marginal gains
- **Deployment vs. CI:** Probably not worth setting up complex CI/CD—clarify priorities
- Don't over-engineer for extra points if it diverts focus

### Easy Wins (High ROI)

- Project management (GitHub issues)
- Docker containerization
- Project management tools (boards, milestone tracking)

---

## Action Items

- [ ] **Research & Literature** — Find best-practice compression starting approaches, send to TA
- [ ] **Clarify Training Split** — Confirm 80/20 person-wise train/test strategy with TA
- [ ] **Finalize Repo Strategy** — Decide: simple linters or skip CI/CD entirely
- [ ] **Timeline Review** — Have TA review updated timeline (with realism check on training)
- [ ] **Schedule Weekly Check-ins** — Await TA availability; aim for weekly meetings

---

## General Approach (TA's Meta-Advice)

1. **Identify complexities early** — compression, novel techniques, etc.
2. **Prioritize accordingly** — don't let peripherals (fancy CI, complex deployment) dominate effort
3. **Plan buffer time** — timeline can flex after week 3–4 if needed
4. **Iterate on bottlenecks first** — solve hard problems early, easy wins later

---

## Notes

- TA's experience: previous cohorts struggled with scope creep on infrastructure and tooling; emphasizes shipping a working model first
