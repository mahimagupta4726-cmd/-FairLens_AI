<<<<<<< HEAD
# ⚖ FairLens AI — Bias Autopsy Engine

> **Detect, Explain, and Fix AI Bias in Hiring, Loans & Healthcare — Instantly.**

![Hackathon Edition](https://img.shields.io/badge/Hackathon-Edition-6c5ce7?style=for-the-badge)
![Fairness Metrics](https://img.shields.io/badge/Fairness_Metrics-12-00b894?style=for-the-badge)
![XAI](https://img.shields.io/badge/XAI-Powered-fd79a8?style=for-the-badge)
![Detection Accuracy](https://img.shields.io/badge/Detection_Accuracy-97%25-fdcb6e?style=for-the-badge)

---

## 🚨 The Problem

Every day, AI systems make thousands of decisions — who gets hired, who gets a loan, who receives medical care. Most of these systems are **biased against gender, race, age, and zip code** — and almost no one can see it.

- A Black applicant with a score of **76** gets rejected. A White applicant with **65** gets approved.
- A woman in healthcare gets deprioritized by a risk model trained on biased historical data.
- A loan applicant in ZIP code 60622 gets denied — not because of their creditworthiness, but their neighborhood.

Existing tools require data science expertise, provide no actionable explanations, and offer no way to fix what they find. **FairLens AI changes that.**

---

## 💡 Solution

**FairLens AI** is a **Bias Autopsy Engine** — a fairness intelligence platform that:

1. **Detects** hidden discrimination using 12 industry-standard fairness metrics
2. **Explains** it with XAI (SHAP-equivalent feature attribution) at the individual level
3. **Fixes** it with AI-generated remediation steps and before/after comparison

Upload any CSV dataset. Get a legally grounded, actionable bias report in seconds — **no ML expertise required.**

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 **Fairness Dashboard** | Composite Fairness Score (0–100) with Demographic Parity, Equalized Odds, and Disparate Impact |
| 🧠 **Explainable AI (XAI)** | SHAP-style feature attribution — see exactly which factors drive unfair decisions |
| 🔍 **Individual Decision Audit** | Click any person's record to see why they were rejected and what counterfactual would have changed the outcome |
| 📈 **Before vs. After Report** | Side-by-side and radar chart comparison showing fairness improvement (e.g., 42 → 84) after debiasing |
| 🔴 **Live Bias Monitor** | Real-time streaming decision feed — catch bias as it happens, not after the damage is done |
| 🛠 **AI Remediation Fixes** | Ranked, specific fixes: reweighting, threshold adjustment, feature removal — each with projected impact |
| ⚖ **Legal Compliance Layer** | Auto-flags violations of EEOC, OFCCP 80% rule, Fair Housing Act, and ACA equity requirements |

---

## 🖥 Demo Walkthrough

### Step 1 — Upload Dataset
- Drag & drop a CSV/JSON file, or pick a built-in sample
- Datasets available: **Hiring** (1,200 applicants), **Loans** (3,500 applications), **Healthcare** (890 patients)
- System auto-detects sensitive attributes: gender, race, age, zip code, income

### Step 2 — Run Fairness Analysis
- Click **"Run Fairness Analysis →"**
- Engine runs: Feature Scan → Bias Detection → Metrics → XAI Report

### Step 3 — Read the Dashboard
- **Fairness Score: 42/100** — Critical
- **Disparate Impact: 0.58** — Legally Actionable (below OFCCP 80% threshold)
- **Demographic Parity: 0.61** — Below threshold
- Bias severity bars, active alerts, and approval-rate-by-group chart

### Step 4 — Explainability
- XAI panel shows which features push decisions (blue = positive, red = negative)
- Click any row in the audit table → full decision explainer modal with counterfactual

### Step 5 — Fix It
- View **Bias Report** tab: Before (42) vs. After (84) — +42 pts improvement
- AI-suggested fixes: reweighting, threshold calibration, feature removal
- Legal compliance checklist auto-generated

### Step 6 — Monitor Live
- **Live Monitor** tab: 142 decisions/min streaming in real time
- Fairness trend chart updates every 1.5 seconds
- Instant flagging: `✓ FAIR` / `⚠ BIAS` / `? REVIEW`

---

## 🛠 Tech Stack

```
Frontend          →  Vanilla HTML5 / CSS3 / JavaScript (zero build step)
Visualizations    →  Chart.js v4.4.1 (bar, line, radar charts)
Fonts             →  Syne (headings) + JetBrains Mono (data)
Fairness Engine   →  Custom JS — Demographic Parity, Equalized Odds,
                      Disparate Impact, Counterfactual Fairness (12 metrics)
XAI Layer         →  SHAP-equivalent feature attribution (in-browser)
Data Input        →  CSV / JSON drag-and-drop, up to 50MB
Hosting           →  Any static host (GitHub Pages, Vercel, Netlify)
```

**Future Scope:**
- Python + FastAPI backend with real SHAP (shap library)
- IBM AIF360 / Fairlearn integration
- Model retraining pipeline with debiasing algorithms
- REST API for integration with existing ML pipelines
- Role-based access for compliance teams

---

## 📁 Project Structure

```
fairlens-ai/
│
├── index.html          # Full application (single-file prototype)
├── README.md           # This file
│
└── (future)
    ├── backend/        # FastAPI + Python fairness engine
    ├── data/           # Sample datasets (hiring, loans, healthcare)
    └── models/         # Debiasing algorithms
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/fairlens-ai.git

# Navigate into the folder
cd fairlens-ai

# Open in browser — no install needed
open index.html
```

> ✅ **No dependencies. No build step. No server required.**  
> Just open `index.html` in any modern browser and it works.

---

## 📊 Fairness Metrics Explained

| Metric | What It Measures | Legal Threshold |
|---|---|---|
| **Demographic Parity** | Approval rate ratio between groups | ≥ 0.80 |
| **Equalized Odds** | Equal TPR + FPR across groups | ≥ 0.80 |
| **Disparate Impact** | OFCCP 80% rule compliance | ≥ 0.80 |
| **Calibration** | Prediction accuracy per group | ≥ 0.85 |
| **Individual Fairness** | Similar people → similar outcomes | ≥ 0.75 |
| **Counterfactual Fairness** | Would outcome change if identity changed? | — |

---

## 🌍 Impact

> *"In the US alone, biased AI costs underrepresented groups an estimated $100B+ annually in lost hiring and lending opportunities."*

FairLens is designed for:
- **HR & Talent Teams** — audit resume screening models before legal exposure
- **Banks & Lenders** — detect redlining patterns in credit scoring
- **Hospitals & Insurers** — identify inequity in treatment prioritization
- **Compliance Officers** — auto-generate regulatory audit reports
- **Regulators & Policymakers** — enforce EU AI Act and US Executive Orders on AI

---

## ⚡ Why FairLens Wins

> *"Every other bias tool tells you there's a problem. FairLens tells you which person was wronged, why it happened, and what their outcome would have been in a fair system."*

- **Not just a score** — individual-level explainability at every decision
- **Not just detection** — remediation steps with projected impact
- **Not just a report** — live monitoring for production AI systems
- **Not just for engineers** — designed for compliance teams, HR, and executives

---

## 👤 Team

| Name | Role |
|---|---|
| Mahima Gupta | Full Stack + Fairness Engine |
| Shreya Yadav | UI/UX design and prototype |
|Zara Alam | Data Science + XAI Visualization |

---

## 📄 License

MIT License — free to use, modify, and build upon.

---

<div align="center">

**Built with ⚖ for fairness, accountability, and impact.**

*FairLens AI · Hackathon Edition · v1.0.0*

</div>
=======
# FairlensAI
FairLens AI — Real-time algorithmic bias detection platform. Upload any hiring, loan, or healthcare dataset and instantly measure Demographic Parity, Disparate Impact &amp; Equalized Odds across gender, race, age, and religion. Built with Python, Flask &amp; Chart.js. EEOC 80% Rule compliance built-in.
>>>>>>> 12b27771261807bd9ce37a711df1f17ffc4f64ae
