# src/train_phi3_lora.py  (Accelerate CPU-offload + fp16, Windows-safe, $0)
import json, torch, random
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from accelerate import Accelerator

model_id = "./model/phi3"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# 300 train / 100 dev split
data = [json.loads(l) for l in open("data/mini_slicebench_400.jsonl")]
random.shuffle(data)
train = data[:300]
dev   = data[300:400]

def fmt(example):
    code, line, slice_txt = example["code"], example["line"], "\n".join(example["slice"])
    prompt = f"### Code:\n{code}\n### Slice for line {line}:\n{slice_txt}\n### End"
    tokens = tokenizer(prompt, truncation=True, max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()   # causal LM loss
    return tokens

train_ds = Dataset.from_list([fmt(ex) for ex in train])
dev_ds   = Dataset.from_list([fmt(ex) for ex in dev])

# fp16 model + gradient checkpointing (no device_map)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
model.gradient_checkpointing_enable()
model = model.cuda()          # explicit CUDA placement

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

accelerator = Accelerator(
    mixed_precision=None,      # <-- was "fp16", now None
    cpu=False
)
model, train_ds, dev_ds = accelerator.prepare(model, train_ds, dev_ds)

training_args = TrainingArguments(
    output_dir="./adapter_phi3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to=None,               # <-- already here
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    load_best_model_at_end=False,
    metric_for_best_model=None,
    greater_is_better=None,
    ignore_data_skip=False,
    skip_memory_metrics=True,
    logging_dir=None,             # <-- no TB
    run_name=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    tokenizer=tokenizer,
)

# let Accelerator handle the optimizer/scheduler
trainer = accelerator.prepare(trainer)
trainer.train()
trainer.save_model("./adapter_phi3")
print("Adapter saved → adapter_phi3/")