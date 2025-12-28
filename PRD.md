# Product Requirements Document (PRD)
## Saree Virtual Try-On — Round 1 Assignment

## 1. Objective
Build a web application that takes a **floor-shot image of a saree** and generates a **photorealistic image of a model wearing the exact same saree**.

The generated output must preserve:
- Main body design
- Pallu (aanchal)
- Border / lace
- Original color tones

The result must be **consistent across 10 consecutive generations**, not a one-off success.

---

## 2. Scope

### In Scope
- Single saree image input
- Image generation using Gemini (Nano Banana Pro / Gemini Image)
- Deterministic generation pipeline
- Automated and manual evaluation
- 10-run reproducibility test

### Out of Scope
- Model training or fine-tuning
- Batch uploads
- User authentication
- Production UI polish

---

## 3. Success Criteria
- 10/10 consecutive generations preserve saree identity
- No hallucinated motifs or colors
- Automated metrics pass for all runs
- Manual QA confirms fidelity

---

## 4. User Flow
1. Upload saree floor-shot image
2. System segments saree and parts
3. Generate model image wearing saree
4. Run 10x reproducibility test
5. View results and pass/fail summary

---

## 5. Constraints
- Image generation models are probabilistic
- Consistency must be enforced by system design
- Determinism is required for evaluation

---

## 6. Deliverables (Round 1)
- Working Next.js app
- Deterministic generation pipeline
- Evaluation results for 10-run test
- Documentation explaining consistency enforcement
