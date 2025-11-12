# Based on https://huggingface.co/Rostlab/ProstT5
# AA-->3Di translation (folding)

import os
import re
import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

cuda_device = "cuda:0"
model_name = "Rostlab/ProstT5"

# Set device (use GPU if available)
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
# model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(device)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    use_safetensors=True  # Force safetensors format (avoids torch version restriction)
).to(device)

# Use half-precision on GPU for efficiency (V100 supports fp16)
if device.type == 'cuda':
    model.half()
else:
    model.full()

# Prepare protein (AA) sequences as a list.
# AA sequences are expected to be upper-case ("PRTEINO" below)
# 3Di sequences need to be lower-case.
sequences = ["PRTEINO", "SEQWENCE"]
min_len = min(len(s) for s in sequences)
max_len = max(len(s) for s in sequences)

# Replace all rare/ambiguous AAs with X (3Di sequences do not have those) and insert whitespace between all sequences (AAs and 3Di)
sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

# Add pre-fix for AA to 3Di translation
sequences = [ "<AA2fold>" + " " + s for s in sequences]

# Tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(
    sequences,
    add_special_tokens=True,
    padding="longest",
    return_tensors='pt'
).to(device)

# Generation configuration for AA-->3Di translation (folding)
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
with torch.inference_mode():  # a small speedup on GPU as opposed to torch.no_grad()
    translations = model.generate(
        ids.input_ids,
        attention_mask=ids.attention_mask,
        max_length=max_len,     # max length of generated text
        min_length=min_len,     # min length of generated text
        early_stopping=True,    # stop early if end-of-text token is generated
        num_return_sequences=1, # return only a single sequence
        **gen_kwargs
    )

# Decode and remove whitespace between tokens
decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
structure_sequences = ["".join(ts.split(" ")) for ts in decoded_translations] # predicted 3Di strings

print("Predicted 3Di sequences:", structure_sequences)
print(f'\n✅ Finished {os.path.basename(__file__)}')
