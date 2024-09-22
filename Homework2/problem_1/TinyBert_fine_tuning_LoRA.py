from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb
import time
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding

# Initialize W&B for logging
wandb.init(project="tinybert-lora-finetuning", config={
    "model": "TinyBERT",
    "dataset": "Tweet Eval Sentiment",
    "epochs": 1,
    "batch_size": 16,
})

# Load the TinyBERT tokenizer and model
dataset = load_dataset("tweet_eval", name="sentiment")
tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=3)

# Set up PEFT with LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,        # training, not inferring
    r=8,
    lora_alpha=32,               # Default scaling factor for LoRA
    lora_dropout=0.1
)

# Convert the model to a LoRA model using PEFT
model = get_peft_model(model, peft_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Record total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.config.update({
    "total_params": total_params,
    "trainable_params": trainable_params
})


# Define compute_metrics function calculate accuracy and F1 score
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# Fine-tune TinyBERT using the same procedure as Part 2 and log metrics
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
    ),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
start_time = time.time()

# Train and evaluate
trainer.train()

training_time = time.time() - start_time
wandb.log({"training_time_seconds": training_time})

if torch.cuda.is_available():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"GPU allocated memory: {allocated_memory:.3f} GB")
    print(f"GPU reserved memory: {reserved_memory:.3f} GB")
    print(f"Max GPU allocated memory: {max_allocated:.3f} GB")


metrics = trainer.evaluate(tokenized_datasets["test"])
wandb.log({
    "test_accuracy": metrics["eval_accuracy"],
    "test_f1_score": metrics["eval_f1"],
    "test_loss": metrics["eval_loss"],
})

wandb.finish()
