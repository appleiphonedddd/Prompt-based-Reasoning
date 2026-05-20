<div align="center">

# 🧠 Prompt-based Reasoning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-FF6B35)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**_Does the way you prompt an LLM fundamentally change how well it reasons?_**

A unified framework for benchmarking **6 prompting strategies** across **5+ LLM providers** and **5 reasoning benchmarks** — reproducible, extensible, and research-ready.

</div>

---

## 🤔 Why This Matters

Modern LLMs are powerful — but their reasoning quality is highly sensitive to *how you ask*.

> **Standard prompt:** "What is 4 × 6 + 11 − 8?"
> **CoT prompt:** "Think step by step. What is 4 × 6 + 11 − 8?"
>
> Same model. Same weights. Dramatically different reasoning quality.

The gap widens on harder tasks — multi-step math, logical deduction, creative generation. This project lets you **measure that gap systematically**.

---

## 📋 Prerequisites

| Requirement | Minimum Spec |
|---|---|
| OS | Ubuntu 24.04.04 LTS |
| GPU | NVIDIA A40 (CUDA 13.x) |
| Python | 3.11 via Conda |
| Docker | Required for local Ollama inference |

---

## 🗂️ Prompting Strategies

| # | Baseline | Key Idea | Reference |
|---|---|---|---|
| 1 | **Standard** | Direct question → answer | [Brown et al., NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) |
| 2 | **Zero-Shot CoT** | "Let's think step by step" | [Kojima et al., NeurIPS 2022](https://openreview.net/pdf?id=e2TBb5y0yFf) |
| 3 | **Tree-of-Thought (ToT)** | Explore & evaluate multiple reasoning paths | [Yao et al., NeurIPS 2023](https://openreview.net/pdf?id=5Xc1ecxO1h) |
| 4 | **Graph-of-Thought (GoT)** | Non-linear thought graphs with merging | [Besta et al., AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29720) |
| 5 | **Buffer-of-Thought (BoT)** | Retrieve & reuse thought templates | [Yang et al., NeurIPS 2024](https://openreview.net/pdf?id=ANO1i9JPtb) |
| 6 | **Reversal-of-Thought (RoT)** | Preference-guided reverse reasoning warm-up | [Yuan et al., ACL 2025](https://aclanthology.org/2025.acl-long.955.pdf) |

---

## 🏗️ Architecture

![Logo](./img/architecture.png)

---

## 🗺️ Support Matrix

### LLM Providers

| Provider | Model Examples | Inference |
|---|---|---|
| Alibaba | `qwen2.5:32b`, `qwen2.5:14b`, `qwen3:8b` | Local via Ollama |
| Meta | `llama3.1:8b`, `llama3.1:70b`, `llama3.3:70b` | Local via Ollama |
| OpenAI | `gpt:gpt-4o`, `gpt:gpt-4o-mini` | Cloud API |
| Google | `gemini:gemini-2.0-flash`, `gemini:gemini-1.5-pro` | Cloud API |
| DeepSeek | `deepseek:deepseek-chat` | Cloud API |

---

## ⚡ Quick Start

### Step 1 — Create Environment

```bash
conda env create -f env.yaml
conda activate Prompt
```

### Step 2a — Local Model (No API Key Needed)

```bash
# First-time GPU + Ollama setup
./setup_ollama_gpu.sh

# Pull a model
docker exec -it ollama ollama run qwen2.5:32b

# Remove a model
docker exec -it ollama ollama rm qwen2.5:32b

# Run your first evaluation
export API_KEY="ollama"
python main.py --model qwen2.5:32b --baseline zerocot --benchmark gameof24
```

### Step 2b — Cloud Model

```bash
export API_KEY="your_key_here"
python main.py --model gpt:gpt-4o-mini --baseline zerocot --benchmark mgsm
```

---

## 📐 Full Usage

```bash
python main.py \
  --model     <model>     \   # e.g. qwen2.5:3b, gpt:gpt-4o
  --baseline  <baseline>  \   # standard | zerocot | rot | tot | bot | got
  --benchmark <benchmark> \   # gameof24 | mgsm | bigbenchhard | sonnetwriting | programmingpuzzles
  [--num_runs N]              # number of problems to evaluate (default: all)
```

### Examples by Prompting Strategy

**Standard — direct baseline**
```bash
python main.py --model qwen2.5:32b --baseline standard --benchmark gameof24
```

**Zero-Shot Chain-of-Thought**
```bash
python main.py --model qwen2.5:32b --baseline zerocot --benchmark mgsm
```

**Tree-of-Thought — breadth-first reasoning search**
```bash
python main.py --model qwen2.5:32b --baseline tot --benchmark gameof24 --tot_n_generate 2 --tot_n_evaluate 1 --tot_breadth 2  --tot_max_steps  1
```

**Graph-of-Thought — non-linear thought graphs**
```bash
python main.py --model qwen2.5:32b --baseline got --benchmark gameof24 --got_branches 3 --got_keep 1 --got_refine 0
```

**BigBenchHard — 27 task categories**
```bash
python main.py --model qwen2.5:32b --baseline zerocot --benchmark bigbenchhard --bigbenchhard_task geometric_shapes
```

<details>
<summary><strong>All 27 BigBenchHard task options</strong></summary>

| Category | `--bigbenchhard_task` |
|---|---|
| Boolean / Yes-No | `boolean_expressions` `causal_judgement` `formal_fallacies` `navigate` `sports_understanding` `web_of_lies` |
| Multiple Choice | `date_understanding` `disambiguation_qa` `geometric_shapes` `hyperbaton` `logical_deduction_3` `logical_deduction_5` `logical_deduction_7` `movie_recommendation` `penguins_in_a_table` `reasoning_about_colored_objects` `ruin_names` `salient_translation_error_detection` `snarks` `temporal_sequences` `tracking_shuffled_objects_3` `tracking_shuffled_objects_5` `tracking_shuffled_objects_7` |
| Numeric | `multistep_arithmetic_two` `object_counting` |
| Free-form | `dyck_languages` `word_sorting` |

</details>

---

## 🔧 Extending the Framework

The framework is built on a **Registry Pattern** — adding new components takes 2 steps each.

### Add a New Model

```python
# 1. Create models/my_model.py
from models.base import BaseLLM, LLMResponse

class MyModelClient(BaseLLM):
    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        return LLMResponse(content="...", model_name=self.model,
                           input_tokens=0, output_tokens=0)

# 2. Register in main.py
MODEL_REGISTRY["mymodel"] = MyModelClient
```

### Add a New Baseline

```python
# 1. Create baseline/MyMethod/my_method.py
from baseline.basebaseline import BaseBaseline, BaselineResponse

class MyMethodBaseline(BaseBaseline):
    def run(self, question: str, **kwargs) -> BaselineResponse:
        return self.create_response(final_answer="...", reasoning_trace="...")

# 2. Register in main.py
BASELINE_REGISTRY["mymethod"] = (MyMethodBaseline, lambda a: dict(...))
```

### Add a New Benchmark

```python
# 1. Create benchmark/MyBench/mybench.py
from benchmark.datasetbase import DatasetBase

class MyBenchmark(DatasetBase):
    def load_dataset(self): ...
    def get_problem(self, index): ...
    def evaluate_answer(self, prediction, ground_truth): ...

# 2. Register in benchmark/__init__.py
DATASET_REGISTRY["mybench"] = (MyBenchmark, lambda _: {})
```

