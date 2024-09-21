import time
import torch
import wandb
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

wandb.init(project="sentiment-classification", config={
    "model": "roberta-base",
    "dataset": "Tweet Eval Sentiment",
    "epochs": 1,
    "batch_size": 16,
})

dataset = load_dataset("tweet_eval", name="sentiment")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Record total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.config.update({
    "total_params": total_params,
    "trainable_params": trainable_params
})

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True, 
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="wandb",
)

# Define compute_metrics function calculate accuracy and F1 score
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

start_time = time.time()

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
