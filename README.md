# RiskRadar - Narrative Risk Triage

A Streamlit prototype that answers: **"For a chosen entity, what are the top narratives right now, how risky are they (0-100), and why — with evidence?"**

Built for the Storyful Lead Data Scientist technical challenge.

## Architecture

```
┌──────────────────────────────────────┐
│  Streamlit App (app.py)              │
├──────────────────────────────────────┤
│  LLM Layer (src/llm.py)             │
│  ├─ Primary: AWS Bedrock (Claude)    │
│  └─ Fallback: local (no LLM needed) │
├──────────────────────────────────────┤
│  Core Pipeline                       │
│  ├─ Entity Resolution (src/entity_resolution.py)   │
│  ├─ Narrative Clustering (src/narrative_clustering.py) │
│  └─ Risk Scoring (src/risk_scoring.py)             │
├──────────────────────────────────────┤
│  Data Layer (src/data_loader.py)     │
│  Feedback (src/feedback.py)          │
└──────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.12+
- (Optional) AWS credentials configured for Bedrock in `eu-west-1`

### Setup

```bash
# Clone and enter the project
cd riskradar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env if you have AWS Bedrock access (see note below)
```

> **No API keys required by default.** The app runs fully locally using sentence-transformers
> and rule-based methods. AWS Bedrock (Claude) is an optional enhancement for better narrative
> summaries and ambiguous entity resolution. To enable it, set `USE_LLM=true` in `.env` and
> configure AWS credentials (`aws configure`) with Bedrock access in `eu-west-1`.

### Place data files

Put the provided data files in the `data/` directory:
```
data/
  posts.jsonl
  authors.csv
  entities_seed.csv
```

### Run the app

```bash
# Optional: pre-compute entity resolution for fast startup
python precompute.py

# Start the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **First run note:** The first startup downloads the sentence-transformer model (`all-MiniLM-L6-v2`, ~80MB)
> and processes all 1000 posts for entity resolution. This takes ~30-60 seconds. Running `python precompute.py`
> first caches the results so the app starts instantly on subsequent runs.

### Run tests

```bash
python -m pytest tests/ -v
```

## How It Works

### 1. Entity Resolution (`src/entity_resolution.py`)

Layered strategy (cheapest first):
1. **Exact match**: Case-insensitive substring match with word boundaries against canonical names + aliases
2. **Fuzzy match**: `rapidfuzz` token_set_ratio for near-matches (threshold: 80)
3. **Embedding match**: `sentence-transformers` cosine similarity for semantic matches (threshold: 0.55)
4. **LLM-assisted**: Bedrock Claude for ambiguous/low-confidence cases (optional)

Each match produces: `{entity_id, mention_text, confidence, resolution_method}`

### 2. Narrative Clustering (`src/narrative_clustering.py`)

- Sentence embeddings via `all-MiniLM-L6-v2`
- Agglomerative clustering with cosine distance (auto-determines cluster count)
- Titles/summaries: LLM-generated (Bedrock) or extractive fallback (most-engaged post)
- Taxonomy labels: LLM or keyword-based classification

### 3. Risk Scoring (`src/risk_scoring.py`)

Weighted sum of 5 normalised drivers (0-100 each):

| Driver | Weight | Rationale |
|--------|--------|-----------|
| Language Risk Signals | 30% | Content determines reputational harm |
| Volume & Velocity | 20% | Posting rate and acceleration |
| Engagement & Amplification | 20% | Likes, shares, comments, views |
| Author Influence | 15% | Follower-weighted reach |
| Virality Indicators | 15% | Share ratios, cross-platform spread |

Each driver provides a human-readable explanation. Confidence is based on sample size and signal consistency.

### 4. Feedback Capture

- Per-post: flag incorrect entity matches, select correct entity
- Per-narrative: adjust risk score, add notes
- All feedback persisted to `feedback.jsonl`

## Output Artifacts

| Artifact | Location |
|----------|----------|
| Feedback | `feedback.jsonl` |
| Output data | `output/` directory |
| Cache | Streamlit's built-in cache |

## Configuration

All configuration in `src/config.py` and `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_LLM` | `true` | Enable Bedrock LLM calls |
| `AWS_REGION` | `eu-west-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-haiku-20240307-v1:0` | Bedrock model |
| `FUZZY_MATCH_THRESHOLD` | 80 | Minimum rapidfuzz score |
| `EMBEDDING_SIMILARITY_THRESHOLD` | 0.55 | Minimum cosine similarity |

## LLM Usage & Guardrails

**Where LLM is used:**
- Entity resolution (ambiguous cases only, after rule-based layers fail)
- Narrative title and summary generation
- Risk taxonomy classification

**Guardrails:**
- LLM is never the sole resolver — always layered after deterministic methods
- JSON parsing with fallback on malformed responses
- Confidence capped at 0.95 for LLM matches (never claims certainty)
- Graceful degradation: app works fully without LLM access
- All LLM calls logged for auditability

**Cost/latency:**
- Uses Claude Haiku (fastest, cheapest Bedrock model)
- LLM only called for posts that fail rule-based resolution (~minority of posts)
- Results cached by Streamlit to avoid repeated calls
