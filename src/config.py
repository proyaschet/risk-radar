"""Central configuration for RiskRadar."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FEEDBACK_PATH = PROJECT_ROOT / "feedback.jsonl"

POSTS_PATH = Path(os.getenv("POSTS_PATH", DATA_DIR / "posts.jsonl"))
AUTHORS_PATH = Path(os.getenv("AUTHORS_PATH", DATA_DIR / "authors.csv"))
ENTITIES_PATH = Path(os.getenv("ENTITIES_PATH", DATA_DIR / "entities_seed.csv"))

# AWS Bedrock
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

# Entity resolution
FUZZY_MATCH_THRESHOLD = 80  # rapidfuzz score threshold
EMBEDDING_SIMILARITY_THRESHOLD = 0.55

# Clustering
MIN_CLUSTER_SIZE = 3
MAX_NARRATIVES = 10

# Risk scoring weights (sum to 1.0)
RISK_WEIGHTS = {
    "volume_velocity": 0.20,
    "engagement": 0.20,
    "author_influence": 0.15,
    "language_risk": 0.30,
    "virality": 0.15,
}

# Risk taxonomy
RISK_TAXONOMY = {
    "regulatory_compliance": "Regulatory / Compliance - alleged breaches, fines, investigations, misconduct claims",
    "financial_integrity": "Financial Integrity - fraud, money laundering, market manipulation, mis-selling",
    "customer_harm": "Customer Harm - poor treatment, unfair practices, discrimination, widespread complaints",
    "data_cyber": "Data / Cyber - breach claims, leaks, ransomware, insecure systems",
    "operational_resilience": "Operational Resilience - outages, service failure, systemic disruption",
    "executive_misconduct": "Executive / Employee Misconduct - leadership scandal, harassment, unethical behaviour",
    "misinfo_manipulation": "Misinformation / Manipulation - coordinated campaigns, synthetic or misleading media narratives",
}

# Ensure output dir exists
OUTPUT_DIR.mkdir(exist_ok=True)
