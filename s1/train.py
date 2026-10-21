"""S1 SFT: batch_rebalance + control-token embedding init. Official weights = checkpoint-300."""
import json
import os

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
HF_USER = os.environ["HF_USER"]
MAX_LEN = 4096
CONTROL = [
    "<|im_start|>", "<|im_end|>", "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
]
HUB_CKPTS = f"{HF_USER}/qwen3-1.7b-rufus-s1-sft-v4-ckpts"
DATA = f"{HF_USER}/rufus-s1-sft-data-v2"


def init_control_embeddings(model, tok):
    E = model.get_input_embeddings().weight
    assert model.lm_head.weight.data_ptr() == E.data_ptr(), "expected tied embeddings"
    with torch.no_grad():
        ref = E[tok.convert_tokens_to_ids("<think>")].clone()
        for t in CONTROL:
            tid = tok.convert_tokens_to_ids(t)
            assert torch.equal(E[tid], ref), f"{t} already distinct; do not overwrite"
            sub = tok(t, add_special_tokens=False, split_special_tokens=True).input_ids
            if sub == [tid]:
                sub = sum((tok(s, add_special_tokens=False).input_ids for s in (t[:1], t[1:-1], t[-1:])), [])
            assert tid not in sub and len(sub) > 1, (t, sub)
            E[tid] = E[sub].mean(0)
            print(f"{t:17s} {tid} <- {tok.convert_ids_to_tokens(sub)}  norm {E[tid].float().norm():.3f}")
        a, b, c = (E[tok.convert_tokens_to_ids(t)].float() for t in ["<think>", "</think>", "<|im_end|>"])
        print("cos think/close", float(F.cosine_similarity(a, b, dim=0)))
        print("cos think/im_end", float(F.cosine_similarity(a, c, dim=0)))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.eos_token = "<|im_end|>"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = load_dataset(DATA, split="train")

    assert torch.cuda.memory_allocated() == 0, "GPU not empty: restart kernel"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    init_control_embeddings(model, tok)

    wandb.finish()
    args = SFTConfig(
        output_dir="/workspace/s1-sft-v4", run_name="s1-sft-v4-qwen3-1.7b",
        num_train_epochs=1, per_device_train_batch_size=8, gradient_accumulation_steps=4,
        learning_rate=2e-5, lr_scheduler_type="cosine", warmup_steps=0.05,
        weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95, max_grad_norm=1.0,
        bf16=True, gradient_checkpointing=True,
        train_sampling_strategy="batch_rebalance",
        max_length=MAX_LEN, packing=False, completion_only_loss=True, eos_token="<|im_end|>",
        logging_steps=5, save_strategy="steps", save_steps=100, save_only_model=True,
        push_to_hub=True, hub_model_id=HUB_CKPTS, hub_strategy="all_checkpoints", hub_private_repo=True,
        report_to="wandb", dataset_num_proc=4, seed=0,
    )
    trainer = SFTTrainer(
        model=model, args=args, processing_class=tok,
        train_dataset=train_ds.remove_columns(["category", "think"]),
    )
    trainer.train()
    log_path = "/workspace/s1-sft-v4/log_history.json"
    json.dump(trainer.state.log_history, open(log_path, "w"))
    HfApi().upload_file(path_or_fileobj=log_path, path_in_repo="log_history.json", repo_id=HUB_CKPTS)


if __name__ == "__main__":
    main()
