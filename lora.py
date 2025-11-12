# Based on ...
# LoRA fine-tuning of ProstT5 for AA-->3Di translation (folding)

import os
import re
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

model_name = "Rostlab/ProstT5"
cuda_device = "cuda:0"

# Set device (multi-GPU via accelerate later if needed)
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    use_safetensors=True,  # Force safetensors format (avoids torch version restriction)
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# Dummy viral-like data: AA sequences (uppercase, spaced) and corresponding 3Di (lowercase, spaced)
# In reality, extract AA from UniProt/FASTA, 3Di from Foldseek on BFVD PDBs
dummy_data = [
    {"aa": "M E T H I O N I N E", "di3": "a v v v v v v v v"},  # Short example
    {"aa": "S E Q U E N C E P R O T E I N", "di3": "d p p p p p p d d d d d d"},  # Longer
    # Add more for a tiny dataset (aim for 100+ in practice)
]

# Create Hugging Face Dataset
dataset = Dataset.from_list(dummy_data)

# Preprocess: Prefix AA with "<AA2fold>", replace rare AAs with X, tokenize source and target
def preprocess(examples):
    # Source: Prefixed spaced AA (uppercase) - don't double-space (aa already has spaces)
    sources = ["<AA2fold> " + re.sub(r"[UZOB]", "X", aa) for aa in examples["aa"]]
    # Target: Spaced 3Di (lowercase, no prefix)
    targets = [di3 for di3 in examples["di3"]]  # Already spaced/lowercase

    # Tokenize sources (NO return_tensors - data collator handles tensor conversion)
    model_inputs = tokenizer(
        sources, max_length=512, padding="longest", truncation=True
    )

    # Tokenize targets (labels) - NO return_tensors
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=512, padding="longest", truncation=True
        ).input_ids
    model_inputs["labels"] = labels

    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# LoRA config: Low-rank adaptation on attention layers (efficient for fine-tuning large T5)
lora_config = LoraConfig(
    r=8,  # Low rank to save VRAM
    lora_alpha=32,
    target_modules=["q", "v", "k", "o"],  # Query/value/key/output projections
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

# Training args: Small for testing; increase for real run
training_args = TrainingArguments(
    output_dir="./prostt5_viral_lora",
    num_train_epochs=3,  # More for convergence
    per_device_train_batch_size=2,  # Adjust based on VRAM (V100 can handle ~8-16)
    gradient_accumulation_steps=4,  # Effective batch 8-64
    learning_rate=1e-4,
    fp16=True,  # Half-precision to fit in VRAM
    save_steps=100,
    logging_steps=10,
    report_to="none",
    remove_unused_columns=False  # Keep for collator
)

# Data collator for seq2seq (handles padding, label masking)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,  # Ignore padding in loss
    pad_to_multiple_of=8 if training_args.fp16 else None  # For AMP efficiency
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train (expect quick on dummy data)
trainer.train()

# Save adapters (only ~MBs, not full model)
model.save_pretrained("./prostt5_viral_lora_adapters")

# Quick inference test post-fine-tune
test_aa = "T E S T S E Q"
test_input = "<AA2fold> " + test_aa  # Already spaced, don't double-space
ids = tokenizer(test_input, return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(
        **ids, max_new_tokens=len(test_aa.split())
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print("Test prediction:", "".join(decoded.split()))  # Remove spaces

print(f'\n✅ Finished {os.path.basename(__file__)}')
