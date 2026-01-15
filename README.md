# 🍌 Mitigating Hallucination in Compact VLMs via Chain-of-Thought

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Model: SmolVLM2](https://img.shields.io/badge/Model-SmolVLM2_2.2B-red)](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Semester Project:** Investigating visual reliability in compact Vision-Language Models (VLMs) and proposing a "Visual Audit" prompting strategy to eliminate hallucination.

---

## 🧐 The Problem
Compact VLMs (like SmolVLM2, <3B params) are efficient for edge deployment but suffer from **"Language Priors."** They often ignore visual evidence in favor of what they *expect* to see (e.g., assuming a banana must have a sticker or be in a bowl).

This project investigates:
1.  **Blindness vs. Hallucination:** Is the model blind, or is it just suggestible?
2.  **The "Purple Banana" Test:** Can it see counter-factual colors? (Result: ✅ Yes)
3.  **The "Phantom Object" Trap:** Can we trick it into inventing objects? (Result: ❌ Yes)
4.  **The Fix:** Using **Chain-of-Thought (CoT)** to force a visual audit.

## 📊 Key Results

| Experiment Type | Sticker (Hallucination) | Apple (Co-occurrence) | Bowl (Visual Ambiguity) |
| :--- | :--- | :--- | :--- |
| **Standard Inference** | ❌ Fails (Invents text) | ❌ Fails (Invents apple) | ❌ Fails (Says "Ceramic") |
| **Defensive Prompt** | ✅ Fixed | ✅ Fixed | ❌ Fails (Still says "Ceramic") |
| **CoT Visual Audit** | ✅ **Fixed** | ✅ **Fixed** | ✅ **Fixed** |

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/vlm-hallucination-project.git](https://github.com/yourusername/vlm-hallucination-project.git)
cd vlm-hallucination-project