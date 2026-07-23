# LLM Refusal Ablation Study (Arbitrary-Rank Ablation on Gemma 4 E2B)


This repository contains the code and evaluation scripts for a research project studying how refusal behavior is encoded in the weights of `google/gemma-4-E2B-it`. Using Arbitrary-Rank Ablation (ARA), I isolate and remove the model's refusal direction, then benchmark the effect on reasoning (ARC-Easy, HellaSwag) and truthfulness (TruthfulQA) to measure what changes and what doesn't.

---

## 📁 Repository Structure

This repository includes the following files to run, test, and benchmark the model:

* `requirements.txt`: The necessary Python dependencies to run the scripts.
* `test_model.py`: A lightweight script for end-users to chat with the model using the Ollama desktop application.
* `comparison.py`: A PyTorch/Transformers script that loads both the base model and the uncensored model side-by-side to compare their outputs on a specific prompt.
* `benchmark.py`: An automated evaluation script using the EleutherAI `lm-eval` harness to scientifically measure logic, reasoning, and truthfulness.
* `README.md`: This documentation file.

---

## ⚙️ Installation & Setup

Before running the evaluation scripts, ensure you have Python installed and a CUDA-compatible GPU (the ablation and testing for this project were performed on an RTX 3080 10GB).

**1. Clone the repository:**
```bash
git clone [https://github.com/Kasper-Bankler/gemma-4-uncensored.git](https://github.com/Kasper-Bankler/gemma-4-uncensored.git)
cd gemma-4-uncensored
```

**2. Install the Python dependencies:**
```bash
pip install -r requirements.txt
```

**3. (Optional) Install Ollama:**
If you want to use the lightweight `test_model.py` script, you must have the [Ollama desktop application](https://ollama.com/) installed and running in the background.

---

## 🚀 Usage Guide

### 1. Simple Inference (`test_model.py`)
The easiest way to interact with the model. This script automatically pulls the GGUF weights from Hugging Face via Ollama and streams the response to your terminal.
```bash
python test_model.py
```

### 2. Side-by-Side Prompting (`comparison.py`)
This script downloads the raw `.safetensors` for both the normal Google model and the uncensored model, and runs inference natively using Transformers. Edit the `PROMPT` variable inside the script to test different adversarial inputs.
```bash
python comparison.py
```

### 3. Scientific Benchmarking (`benchmark.py`)
Runs the standardized EleutherAI Language Model Evaluation Harness. By default, it tests ARC-Easy (Logic), HellaSwag (Common Sense), and TruthfulQA (Adversarial Truthfulness) 
```bash
python benchmark.py
```

---

## 📊 Benchmark Findings

The tests were run zero-shot using a 500-question sample (`limit=500`) to measure logic, reasoning, and susceptibility to false premises.

| Benchmark (Task) | Normal Gemma 4 (Base) | Uncensored Gemma 4 (Trial 71) | Difference |
| :--- | :--- | :--- | :--- |
| **ARC-Easy** (Logic & Science) | 30.00% | 29.00% | - 1.00% |
| **HellaSwag** (Common Sense) | 28.80% | 29.80% | + 1.00% |
| **TruthfulQA (MC2)** (Adversarial) | 46.49% | 45.21% | - 1.28% |

### 📈 Analysis & Takeaways

1. **Logic Preservation (ARC & HellaSwag):** The scores for general reasoning and common sense are virtually identical, resting well within the margin of error. This proves that a low-ablation approach (KL Divergence: `0.1650`) successfully targets the safety filter without causing "brain damage" or semantic drift to the model's core intelligence.
2. **The TruthfulQA Drop:** TruthfulQA is an adversarial benchmark packed with conspiracy theories and common misconceptions. The uncensored model scores slightly worse here, which is the expected outcome. By surgically removing the AI's "refusal vector," it loses its corporate safety instinct to reject false premises, making it more willing to engage with controversial or hallucinated logic.

---

## ⚠️ Disclaimer and Terms of Use

**1. Academic Research Context**
This model was developed exclusively as a university research project to study Representation Engineering, Arbitrary-Rank Ablation (ARA), and the mechanical nature of Large Language Model alignment. It is intended strictly for academic, educational, and research purposes.

**2. Removed Safety Guardrails**
Because this model has been intentionally abliterated (uncensored) at the matrix level, it no longer adheres to standard safety guidelines. It can and will generate content that may be considered offensive, harmful, explicit, or dangerous if prompted to do so. 

**3. No Liability for Misuse**
By downloading or interacting with this model, you assume full responsibility for how you use it. The creator of this model assume absolutely no liability for any consequences, damages, or harm resulting from the use of this model or the content it generates. You are strictly prohibited from using this model to facilitate illegal acts, cyberattacks, or real-world harm.

**4. Factual Inaccuracy and Hallucinations**
This is a small 2-Billion parameter model. Without its standard RLHF training, it is highly prone to severe semantic drift and aggressive hallucinations when pushed outside its core knowledge domains. **Do not** rely on this model for factual accuracy, and under no circumstances should it be used for medical, legal, or financial advice. 

**Use at your own risk.**
