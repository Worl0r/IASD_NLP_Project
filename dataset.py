def preprocessing_fn(x, tokenizer):
    x["input_ids"] = tokenizer(
        x["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    return x
