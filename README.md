# Leaving, Staying, or Living There? Duration Commonsense Reasoning with Human-LLM Co-annotations

Code and data for the IEEE BigData 2026 paper *Leaving, Staying, or Living There? Duration Commonsense Reasoning with Human-LLM Co-annotations* (Chenyang Yu, Zhaomin Xiao, Yan Huang, and Eduardo Blanco).

Given a geotagged tweet (text and image), the task is to infer how long the author will stay in the geotagged city after posting: `0-24 hours`, `1-30 days`, or `more than 30 days`. The corpus is co-annotated by crowdworkers and LLMs, and we evaluate pretrained language models (PTLMs), text-only LLMs, and vision-language models (VLMs) with multi-turn Chain-of-Thought prompting.

## Repository layout

| Path | Contents |
|---|---|
| `annotations/new_annotation.jsonl` | Adjudicated labels for the 1,095 test instances (`instance_id`, `adjudicated_label`) |
| `annotations/human_annotations/` | Per-batch human annotations (21 batches of 500), with per-annotator labels |
| `data_split.json` | Train / dev / test instance ids (3,500 / 876 / 1,095) |
| `experimental_results/responses/` | Saved raw responses of every evaluated LLM/VLM on the test set |
| `experimental_results/results/` | Metrics computed from the saved responses (`response_test.py` output) |
| `experimental_results/predictions/` | Saved PTLM predictions |
| `model_architecture/` | PTLM baseline architectures (selected via `main.py -model ...`) |
| `utils/` | Annotation pipeline (`annotation_filter/`), training helpers (`learning_helper/`), tweet filtering (`tweet_filter/`) |
| `llm_*.py` | Scripts that query each LLM/VLM family (see below) |
| `main.py` | PTLM training and evaluation |
| `response_test.py` | Recompute the evaluation metrics from the saved responses |
| `other/` | `MACE.jar` (label adjudication), a sample AMT batch file, and the tweet-rendering template used for figures |

**Privacy note:** all MTurk WorkerIds are pseudonymized (`annotator_0001`, `annotator_0002`, ...) with a single consistent mapping across all files. The released files contain no usernames or profile images.

## Data

Tweet text and images are distributed separately:

```
wget https://zhaomin.s3.us-east-2.amazonaws.com/public_projects/duration_final_data.zip
unzip duration_final_data.zip   # creates data/ at the repository root (git-ignored)
```

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

### 1. Recompute the metrics from the saved responses (no API access needed)

```
python response_test.py
```

This reads `annotations/new_annotation.jsonl` and `experimental_results/responses/`, and writes `experimental_results/results/response_test_summary.json`.

### 2. Regenerate the LLM/VLM responses

| Script | Models | Access required |
|---|---|---|
| `llm_openai.py` | GPT-5-mini, Grok 4 | OpenAI-compatible API key (e.g., OpenKey) |
| `llm_gemini.py` | Gemini and Claude families | Google Cloud Vertex AI |
| `llm_llama3.py` | Llama 3.2 (text and vision) | Local GPU + Hugging Face weights |
| `llm_qwen.py` | Qwen 2.5 / Qwen2.5-VL | Local GPU + Hugging Face weights |
| `llm_glm.py` | GLM-4V | Local GPU + Hugging Face weights |

### 3. Train and evaluate the PTLM baselines

```
python main.py -model no_matcher -num_epochs 10 -batch_size 8 -learning_rate 1e-06 -early_stopping 2 -use_image -pooling maxpooling
python main.py --repeated_run --batch_size 64 --learning_rate 1e-06 --num_epochs 1000 --early_stopping 5 --model clip2_TIL
```

## Citation

```bibtex
@inproceedings{yu2026leaving,
  title     = {Leaving, Staying, or Living There? Duration Commonsense Reasoning with Human-LLM Co-annotations},
  author    = {Chenyang Yu and Zhaomin Xiao and Yan Huang and Eduardo Blanco},
  booktitle = {2026 IEEE International Conference on Big Data (BigData)},
  address   = {Phoenix, AZ, USA},
  publisher = {IEEE},
  year      = {2026}
}
```
