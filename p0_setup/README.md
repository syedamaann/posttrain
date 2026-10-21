# P0 — machine + tokenizer

Smoke test before any train/eval. Confirms GPU, bf16, Hub token, and Qwen special-token ids.

```bash
python p0_setup/check_env.py
```

Expect `eos <|im_end|> 151645`, pad `<|endoftext|>` 151643, `<think>` 151667, `</think>` 151668.

On RunPod: set env from `.env.example` on the **pod**, not in git. After `pip install` of `trl`/`datasets`, **restart the kernel** before importing (pyarrow ABI mix).
