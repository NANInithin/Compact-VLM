# 🍌 Mitigating Hallucination in Compact VLMs via Chain-of-Thought

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Model: SmolVLM2](https://img.shields.io/badge/Model-SmolVLM2_2.2B-red)](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Semester Project:** Investigating visual reliability in compact Vision-Language Models (VLMs) and proposing a "Visual Audit" prompting strategy to eliminate hallucination.

---

## 🧐 The Problem

Compact VLMs (like SmolVLM2, <3B params) are efficient for edge deployment but suffer from **"Language Priors."** They often ignore visual evidence in favor of what they *expect* to see (e.g., assuming a banana must have a sticker or be in a bowl).

This project investigates:

1. **Blindness vs. Hallucination:** Is the model blind, or is it just suggestible?
2. **The "Purple Banana" Test:** Can it see counter-factual colors? (Result: ✅ Yes)
3. **The "Phantom Object" Trap:** Can we trick it into inventing objects? (Result: ❌ Yes)
4. **The Fix:** Using **Chain-of-Thought (CoT)** to force a visual audit.

## 📊 Key Results

| Experiment Type | Sticker (Hallucination) | Apple (Co-occurrence) | Bowl (Visual Ambiguity) |
| :--- | :--- | :--- | :--- |
| **Standard Inference** | ❌ Fails (Invents text) | ❌ Fails (Invents apple) | ❌ Fails (Says "Ceramic") |
| **Defensive Prompt** | ✅ Fixed | ✅ Fixed | ❌ Fails (Still says "Ceramic") |
| **CoT Visual Audit** | ✅ **Fixed** | ✅ **Fixed** | ✅ **Fixed** |

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/vlm-hallucination-project.git
cd vlm-hallucination-project
```

### 2. Set up the Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (Optimized for CUDA 12.8)
pip install -r requirements.txt
```

### 3. Generate Test Data

We use a script to generate the adversarial "Purple Banana" and verify the "Real Banana" exists.

```bash
python src/generate_trap_real.py
```

## 🧪 Experiments

### 1️⃣ The "Purple Banana" Test (Modality Check)

Tests if the model actually looks at the image or just guesses colors based on the object name.

```bash
python src/experiment.py
```

**Hypothesis:** If it says "Yellow" for a purple banana, it has Modality Collapse.

**Result:** It correctly identified "Purple/Pink," proving the Vision Encoder is robust.

### 2️⃣ The "Phantom Object" Test (Baseline)

Tests suggestibility by asking about non-existent items (stickers, bowls).

```bash
python src/experiment_phantom.py
```

**Result:** The model hallucinated a sticker text and a ceramic bowl.

### 3️⃣ The "Chain-of-Thought" Fix (Solution)

Implements a 2-step "Visual Audit" prompt:

1. List visible objects.
2. Answer based ONLY on that list.

```bash
python src/experiment_cot.py
```

**Result:** Hallucinations dropped to 0%.

## 📂 Repository Structure

```
├── data/                  # Experiment images (generated locally)
├── src/
│   ├── experiment.py      # Basic adversarial color test
│   ├── experiment_phantom.py # Hallucination probing
│   ├── experiment_fix.py  # Attempt 1: Defensive Prompting
│   ├── experiment_cot.py  # Attempt 2: Chain of Thought (The Solution)
│   └── generate_trap_real.py # Data generation script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 🤝 Acknowledgments

- **Model:** SmolVLM2-2.2B-Instruct by Hugging Face TB.
- **Hardware:** Experiments ran on NVIDIA RTX 4060 Laptop GPU (8GB VRAM).
