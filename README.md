# Trustworthy Explainable Search in Digital Cultural Heritage

## Overview

This project investigates the trustworthiness of AI-generated explanations for visual cultural heritage search.

The case study uses the Digital Documerica collection. A multimodal large language model (Qwen2.5-VL-7B-Instruct) generates multiple explanations for the same historical photograph. The explanations are compared for semantic consistency, evaluated using natural language inference (NLI), and cross-validated against available archival metadata.

A central limitation is that an explanation can be internally consistent while still being factually incorrect. Therefore, consistency is treated as a screening signal rather than proof of truth.

## Research Questions

1. How consistent are multiple independently generated explanations for the same cultural heritage image?
2. Can semantic similarity and NLI identify potentially hallucinated or contradictory claims?
3. To what extent can archival metadata provide additional evidence for evaluating generated claims?
4. How can NLI-based consistency and archival metadata validation be presented as interpretable evidence for explanation reliability?

## Dataset

The analysis uses a subset of the Digital Documerica cultural heritage collection.

- 96 metadata records were initially considered.
- 92 images were successfully retrieved.
- 4 images could not be retrieved because of HTTP 403 responses.
- 5 explanations were generated for each retrievable image.
- Total generated explanations: 460.
- Total segmented sentences: 2,876.

The generated explanation dataset is provided in:

`results/documerica_all_460_explanations_FIXED.csv`

Original cultural heritage images are not redistributed in this repository.

## Explanation Generation

Model: Qwen2.5-VL-7B-Instruct

Five independently sampled explanations were generated for each image.

The model was instructed to describe the image based only on visually observable information, distinguish observations from interpretations, and avoid unsupported claims about dates, locations, names, occupations, historical events, and other facts that cannot be established visually.

The complete generation prompt is available in:

`prompts/explanation_prompt.txt`

## Analysis Pipeline

The analysis consists of the following stages:

1. Sentence segmentation
2. Sentence embedding using Sentence-BERT / MiniLM
3. Pairwise cosine similarity
4. Similarity-based candidate selection
5. Natural language inference using DeBERTa
6. Image-level consistency calculation
7. Claim clustering
8. Claim-field classification
9. Metadata cross-validation
10. Reliability scoring
11. Manual validation and qualitative case studies

## Sentence Similarity

Sentence embeddings contain 384 dimensions.

Within each image, sentence pairs from different explanations were compared using cosine similarity.

At the primary similarity threshold of 0.60, 4,163 sentence pairs were selected for NLI analysis.

Similarity-threshold sensitivity was evaluated at 0.50, 0.60, 0.70, and 0.80.

## Natural Language Inference

The NLI model used was `cross-encoder/nli-deberta-v3-base`.

Candidate sentence pairs were classified as:

- Entailment
- Neutral
- Contradiction

For consistency analysis, neutral pairs were excluded.

The consistency measure is:

`Entailment / (Entailment + Contradiction)`

This measures agreement among comparable generated claims. It does not establish factual truth.

## Claim-Level Analysis

Semantically similar sentences were grouped into claim clusters using a cosine similarity threshold of 0.75.

Sensitivity was tested at 0.70, 0.75, and 0.80.

Claims were also assigned preliminary fields:

- Visual
- Context
- Location
- Date

The field classification is keyword-based and should therefore be interpreted as a preliminary analytical categorization rather than manually validated ground truth.

## Metadata Cross-Validation

Generated claims were compared with available archival metadata, including city, state, date taken, byline, keywords, and exhibit description.

Metadata validation distinguishes between supported, not supported by metadata, unverifiable, and not metadata-checkable claims.

Missing metadata is treated as unavailable evidence rather than evidence that a generated claim is false.

In the current validation output, 280 claims were metadata-checkable. Of these, 276 were classified as unverifiable and 4 as potentially supported; no claims were classified as unsupported by the available metadata.

## Reliability Indicator

The final image-level reliability indicator is based on NLI consistency across independently generated explanations.

For each image, neutral NLI comparisons are excluded, and consistency is calculated as:

`Entailment / (Entailment + Contradiction)`

The resulting score is expressed as a percentage and categorized as:

- Low: < 60%
- Moderate: 60%–<80%
- High: ≥ 80%

For the 92 analyzed images:

- Mean reliability: 81.85%
- Median: 83.77%
- Standard deviation: 16.16 percentage points
- Minimum: 30.77%
- Maximum: 100.00%

Category distribution:

- Low: 10 images
- Moderate: 24 images
- High: 58 images

Archival metadata is used as a separate cross-validation layer rather than as a fixed weighted component because metadata availability is sparse. Missing metadata is treated as unavailable evidence rather than evidence that a generated claim is false.

The reliability score is an analytical measure of cross-explanation consistency, not a calibrated probability that an explanation is factually correct.

## Manual Validation

A manual validation sample of 30 NLI sentence pairs was evaluated.

- 10 true contradictions
- 18 false positives
- 2 ambiguous cases

Excluding the ambiguous cases, the observed precision was approximately 35.71%.

This demonstrates that NLI contradiction detection is useful as a screening mechanism but is not sufficient on its own to identify hallucinations reliably.

## Qualitative Case Studies

Six images were selected for qualitative analysis:

- 0071
- 0039
- 0030
- 0063
- 0018
- 0074

The case-study explanations are provided in:

`results/case_study_explanations.csv`

Contradiction examples are provided in:

`results/case_study_contradictions.csv`

## Repository Structure

```text
trustworthy-explainable-search-documerica/
├── README.md
├── notebooks/
│   └── documerica_trustworthy_explainable_search.ipynb
├── prompts/
│   └── explanation_prompt.txt
├── data/
├── results/
└── figures/
```

## Reproducibility

The main analysis notebook is available in `notebooks/`.

Generated datasets and analysis results are provided in `results/`.

The explanation-generation prompt is provided in `prompts/`.

The original cultural heritage images are not redistributed. Reproduction of the image-generation stage requires obtaining the relevant images from the original Digital Documerica collection and complying with applicable access and redistribution conditions.

## Limitations

- Semantic consistency does not imply factual truth.
- NLI contradiction detection can produce false positives.
- Metadata coverage is sparse for many claims.
- Claim-field classification is preliminary and keyword-based.
- The reliability score is not calibrated as a probability.
- Claim clustering is order-dependent.
- The analysis covers 92 successfully retrieved images rather than the complete 96-record metadata set.
- Some generated explanations may be truncated because of generation-length constraints.

## References

Arnold, T. & Tilton, L. (2024). Explainable Search and Discovery of Visual Cultural Heritage Collections with Multimodal Large Language Models. CHR 2024.

Bai, S. et al. (2025). Qwen2.5-VL Technical Report.

He, P. et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.

Manakul, P., Liusie, A. & Gales, M. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. EMNLP 2023.

Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

## AI Use Disclosure

Generative AI tools were used during this research project for image explanation generation, analysis assistance, code development and debugging, and language editing. The final analytical decisions, interpretation of results, and academic argumentation remain the responsibility of the author.