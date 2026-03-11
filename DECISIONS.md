# Key Decisions & Trade-offs

## Architecture Decisions

### Layered entity resolution (not LLM-first)
**Decision:** Use exact > fuzzy > embedding > LLM resolution layers.
**Rationale:** Most entities match via exact alias lookup (fast, deterministic, auditable). LLM is reserved for genuinely ambiguous cases. This minimises cost and latency while maintaining high recall.
**Trade-off:** May miss creative references or indirect mentions that only an LLM would catch.

### Agglomerative clustering over HDBSCAN
**Decision:** Use scikit-learn's AgglomerativeClustering with cosine distance threshold.
**Rationale:** More predictable behaviour with small datasets (1000 posts). HDBSCAN requires tuning min_samples and min_cluster_size, and can be overly aggressive with noise labelling on small corpora.
**Trade-off:** Less adaptive than HDBSCAN for varying density clusters.

### Weighted linear risk score (not ML classifier)
**Decision:** Explicit weighted sum of 5 normalised signal dimensions.
**Rationale:** Fully explainable, auditable, and tunable. In regulated environments (financial services), stakeholders need to understand exactly how a score was produced. No training data available for a classifier.
**Trade-off:** Less nuanced than a learned model; weights are hand-tuned rather than data-driven.

### Bedrock + local fallback (dual mode)
**Decision:** AWS Bedrock (Claude Haiku) as primary LLM, with full local fallback.
**Rationale:** Bedrock provides managed, production-grade LLM access in eu-west-1. Fallback ensures the app works for anyone without AWS access.
**Trade-off:** Two code paths to maintain; local fallback produces lower-quality summaries.

### Sentence-transformers for embeddings
**Decision:** Use `all-MiniLM-L6-v2` (384-dim) for all embedding tasks.
**Rationale:** Small, fast, good quality for semantic similarity. Single model simplifies deployment.
**Trade-off:** Larger models (e.g., `all-mpnet-base-v2`) would give better quality but slower inference.

## Scoring Rationale

### Weight justification
- **Language risk (30%):** Content is the primary signal for reputational harm. A post about "fraud" is inherently riskier than one with high engagement but benign content.
- **Volume & velocity (20%):** Trending narratives are more urgent. Acceleration matters more than absolute count.
- **Engagement (20%):** Amplification indicates broader awareness and potential media pickup.
- **Author influence (15%):** High-follower accounts amplify reach disproportionately.
- **Virality (15%):** Cross-platform spread and share ratios indicate organic amplification.

### Confidence levels
- **High:** 10+ posts, consistent signals across drivers
- **Medium:** 5-9 posts
- **Low:** <5 posts or highly variable signals

## If I Had More Time (2-4 weeks)

### Week 1: Evaluation & Ground Truth
- Hand-label 200+ posts for entity resolution accuracy
- Create gold-standard narrative clusters for 3-5 entities
- Build automated evaluation pipeline (precision/recall/F1)
- A/B test fuzzy threshold and embedding threshold values

### Week 2: Model Improvements
- Train a fine-tuned NER model on the domain (pharma entities)
- Replace keyword-based taxonomy with zero-shot classification (DeBERTa NLI)
- Add temporal analysis (trend detection, burst detection)
- Implement proper sentiment analysis (not just keyword-based)

### Week 3: Production Hardening
- Add Langfuse for LLM observability and tracing
- Build feedback loop: use collected feedback to retrain/tune
- Add caching layer (Redis) for embeddings and LLM responses
- Implement rate limiting and cost controls for Bedrock
- Add data drift monitoring

### Week 4: Scale & Deploy
- Containerise with Docker
- Deploy on ECS Fargate with ALB
- Add CI/CD pipeline
- Pre-compute embeddings on data ingestion (async pipeline)
- Add user authentication and RBAC
- Build API layer for programmatic access
