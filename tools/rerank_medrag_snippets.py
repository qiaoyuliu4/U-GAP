"""U-GAP runtime components extracted from the research implementation."""

import torch

def score_pairs(model, tokenizer, queries, texts, device, batch_size, max_length):
    logits = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_queries = queries[start : start + batch_size]
        pairs = [[query, text] for query, text in zip(batch_queries, batch_texts)]
        encoded = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            batch_logits = model(**encoded).logits
        if batch_logits.ndim == 2 and batch_logits.shape[1] == 1:
            batch_logits = batch_logits[:, 0]
        elif batch_logits.ndim == 2:
            batch_logits = batch_logits[:, -1]
        logits.extend(float(value) for value in batch_logits.detach().float().cpu().tolist())
    return logits
