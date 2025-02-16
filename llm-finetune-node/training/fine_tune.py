import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load dataset
dataset = load_dataset("json", data_files="../data/data.jsonl")

# Load tokenizer and model
model_name = "mistralai/Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Tokenize dataset
def tokenize_data(example):
    inputs = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    labels = tokenizer(example["response"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return {"input_ids": inputs["input_ids"][0], "attention_mask": inputs["attention_mask"][0], "labels": labels["input_ids"][0]}

tokenized_dataset = dataset.map(tokenize_data, remove_columns=["prompt", "response"])

# LoRA Configuration
lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/fine_tuned_model",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir="../logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# Train model
trainer.train()

# Save model
model.save_pretrained("../models/fine_tuned_model")
tokenizer.save_pretrained("../models/fine_tuned_model")

print("Fine-tuning complete! Model saved to models/fine_tuned_model")
