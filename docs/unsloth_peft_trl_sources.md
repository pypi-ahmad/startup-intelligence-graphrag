# Unsloth, PEFT, and TRL Source Notes

This project uses Unsloth + PEFT + TRL only in the optional domain-adapter stage.

## Why this file exists

The adapter implementation is documentation-driven and intentionally isolated.
This note records the official references used for the current implementation approach.

## Official references

### Unsloth

- Unsloth docs home:
  - https://docs.unsloth.ai/
- Granite-specific support page:
  - https://unsloth.ai/docs/models/ibm-granite-4.1

### TRL (Hugging Face)

- TRL docs home:
  - https://huggingface.co/docs/trl
- SFT Trainer docs:
  - https://huggingface.co/docs/trl/sft_trainer

### PEFT (Hugging Face)

- PEFT docs home:
  - https://huggingface.co/docs/peft
- LoRA package reference:
  - https://huggingface.co/docs/peft/package_reference/lora

## How these tools are applied in this repository

- File: `src/extensions/domain_adapter.py`
- Unsloth:
  - `FastLanguageModel.from_pretrained`
  - `FastLanguageModel.get_peft_model`
- TRL:
  - `SFTConfig`
  - `SFTTrainer`
- PEFT:
  - `PeftModel.from_pretrained`
  - optional adapter merge path

## Scope boundary

These tools are not used in:
- ingestion
- chunking
- vector indexing
- graph construction
- hybrid retrieval
- agentic routing/CRAG loop
- multimodal OCR/vision retrieval

They are used only in the optional generator adaptation stage.
