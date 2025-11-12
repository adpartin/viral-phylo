# Based on https://huggingface.co/Rostlab/ProstT5

import os
import re
import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
# from transformers import TRANSFORMERS_CACHE # for caching models

cuda_device = "cuda:0"
# Set device (use GPU if available)
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Check cache location
# cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
# print(f"Model cache directory: {cache_dir}/hub/models--Rostlab--ProstT5/")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
# model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(device)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Rostlab/ProstT5",
    use_safetensors=True  # Force safetensors format (avoids torch version restriction)
).to(device)

# Use half-precision on GPU for efficiency (V100 supports fp16 well)
if device.type == 'cuda':
    model.half()
else:
    model.full()

# Example AA sequences (uppercase, replace rare AAs with X)
folding_examples = ["PRTEINO", "SEQWENCE"]
min_len = min(len(s) for s in folding_examples)
max_len = max(len(s) for s in folding_examples)

# Preprocess: add spaces between residues
folding_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in folding_examples]

# Add prefix for AA to 3Di translation
folding_examples = ["<AA2fold> " + s for s in folding_examples]

# Tokenize
ids = tokenizer.batch_encode_plus(
    folding_examples,
    add_special_tokens=True,
    padding="longest",
    return_tensors='pt'
).to(device)

# Generation hyperparameters (tuned for diversity and quality)
gen_kwargs = {
    "do_sample": True,
    "num_beams": 3,
    "top_p": 0.95,
    "temperature": 1.2,
    "top_k": 6,
    "repetition_penalty": 1.2,
}

# Translate from AA to 3Di (AA-->3Di) (folding)
# with torch.no_grad():
with torch.inference_mode():  # Provides a small speedup on GPU as opposed to torch.no_grad()
    translations = model.generate(
        ids.input_ids,
        attention_mask=ids.attention_mask,
        max_length=max_len,
        min_length=min_len,
        early_stopping=True,
        num_return_sequences=1,
        **gen_kwargs
    )

# Decode and remove spaces
decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
structure_sequences = ["".join(ts.split(" ")) for ts in decoded_translations]

print("Predicted 3Di sequences:", structure_sequences)
print(f'\n✅ Finished {os.path.basename(__file__)}')
