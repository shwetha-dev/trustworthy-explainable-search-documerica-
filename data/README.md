# Dataset Documentation

## Digital Documerica

This project uses historical cultural heritage photographs from the Digital Documerica collection.

The analysis began with 96 metadata records. Of these, 92 images were successfully retrieved and used for multimodal explanation generation. Four images could not be retrieved because the source returned HTTP 403 responses.

## Dataset Statistics

- Metadata records: 96
- Successfully retrieved images: 92
- Unretrievable images: 4
- Explanations per image: 5
- Total explanations: 460
- Total segmented sentences: 2,876

## Metadata

The available metadata includes fields such as:

- image name
- image ID
- city
- state
- date taken
- keywords
- exhibit description
- byline

Metadata availability is incomplete. Missing metadata is therefore treated as unavailable evidence during validation rather than as evidence that a generated claim is false.

## Generated Explanations

Five independently sampled explanations were generated for each successfully retrieved image using Qwen2.5-VL-7B-Instruct.

The model was instructed to:

- describe only visually observable information,
- distinguish observations from interpretations,
- use cautious language for uncertain interpretations,
- avoid unsupported dates, locations, names, occupations, historical events, and similar claims.

The generation prompt is stored in:

`../prompts/explanation_prompt.txt`

The complete generated explanation dataset is stored in:

`../results/documerica_all_460_explanations_FIXED.csv`

## Image Availability

The original cultural heritage images are not redistributed in this repository.

Users reproducing the image-generation stage should obtain the images from the original Digital Documerica collection and follow the applicable access and usage conditions.

## Scope of Analysis

All downstream analysis in this repository is based on the 92 successfully retrieved images and their 460 generated explanations.
