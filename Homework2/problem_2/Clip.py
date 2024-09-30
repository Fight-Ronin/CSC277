import torch
import torch.nn as nn
import clip
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import wandb

# Initialize W&B
wandb.init(project="vqa-clip", config={
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
})

# Dataset class for loading the VQA data
class CustomVQADataset(Dataset):
    def __init__(self, dataframe, label_map, image_dir="data/images", transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = label_map
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        question = self.dataframe.iloc[idx]['question']
        image_id = self.dataframe.iloc[idx]['image_id']
        answer = self.dataframe.iloc[idx]['answer']
        label = self.label_map[answer]

        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            return None

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return question, image, label

# Define image transformations for CLIP
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# Load CSV files for training, validation, and test sets
train_df = pd.read_csv('data/new_data_train.csv')
val_df = pd.read_csv('data/new_data_val.csv')
test_df = pd.read_csv('data/new_data_test.csv')

# Create label map from unique answers
all_answers = pd.concat([train_df['answer'], val_df['answer'], test_df['answer']])
unique_answers = all_answers.unique()
label_map = {answer: idx for idx, answer in enumerate(unique_answers)}

# Log the label map to W&B
wandb.config.update({"label_map_size": len(label_map)})

# Create Datasets and DataLoaders
train_dataset = CustomVQADataset(dataframe=train_df, label_map=label_map, transform=image_transform)
val_dataset = CustomVQADataset(dataframe=val_df, label_map=label_map, transform=image_transform)
test_dataset = CustomVQADataset(dataframe=test_df, label_map=label_map, transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# Visual Encoder using CLIP
class VisualEncoder(nn.Module):
    def __init__(self, device):
        super(VisualEncoder, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)

    def forward(self, x):
        with torch.no_grad():
            img_embeddings = self.model.encode_image(x)
        return img_embeddings

# Textual Encoder using CLIP
class TextualEncoder(nn.Module):
    def __init__(self, device='cpu', model_name="ViT-B/32"):
        super(TextualEncoder, self).__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.device = device

    def forward(self, sentences):
        if isinstance(sentences, (list, tuple)):
            sentences = clip.tokenize(sentences).to(self.device)
        
        # 确保 sentences 是 Tensor 类型
        if not isinstance(sentences, torch.Tensor):
            raise ValueError(f"Input sentences must be a torch.Tensor, but got {type(sentences)}")

        with torch.no_grad():
            text_embeddings = self.model.encode_text(sentences)

        return text_embeddings

# Combined Model for Multi-Modal Fusion
class CombinedModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=30):
        super(CombinedModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512 + 512, hidden_dim),  # 确保输入的维度是 512 + 512
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_emb, text_emb):
        img_emb = img_emb.to(dtype=torch.float32)  # 强制转换 img_emb 到 float32
        text_emb = text_emb.to(dtype=torch.float32)  # 强制转换 text_emb 到 float32

        concatenated_emb = torch.cat((img_emb, text_emb), dim=-1)
        wandb.log({
            "Image Embedding Dimension": str(img_emb.shape),
            "Text Embedding Dimension": str(text_emb.shape),
            "Concatenated Embedding Dimension": str(concatenated_emb.shape)
        })
        print(f"Image Embedding Dimension: {img_emb.shape}")
        print(f"Text Embedding Dimension: {text_emb.shape}")
        print(f"Concatenated Embedding Dimension: {concatenated_emb.shape}")

        return self.fc(concatenated_emb)

# Training function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = VisualEncoder(device).to(device)
    textual_encoder = TextualEncoder(device=device).to(device)
    combined_model = CombinedModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(combined_model.parameters(), lr=wandb.config.learning_rate)

    for epoch in range(wandb.config.epochs):
        combined_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for questions, images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            text_embeddings = textual_encoder(questions).to(device)
            img_embeddings = visual_encoder(images).to(device)

            outputs = combined_model(img_embeddings, text_embeddings)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"step_loss": loss.item()})
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

    test_loss, test_acc = evaluate(combined_model, test_loader, visual_encoder, textual_encoder, device)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Train and evaluate function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = VisualEncoder(device).to(device)
    textual_encoder = TextualEncoder(device=device).to(device)
    combined_model = CombinedModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(combined_model.parameters(), lr=wandb.config.learning_rate)

    for epoch in range(wandb.config.epochs):
        combined_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for questions, images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            text_embeddings = textual_encoder(questions).to(device)
            img_embeddings = visual_encoder(images).to(device)

            outputs = combined_model(img_embeddings, text_embeddings)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"step_loss": loss.item()})
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

    test_loss, test_acc = evaluate(combined_model, test_loader, visual_encoder, textual_encoder, device)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Evaluate function remains the same.
def evaluate(model, data_loader, visual_encoder, textual_encoder, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for questions, images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            text_embeddings = textual_encoder(questions).to(device)
            img_embeddings = visual_encoder(images).to(device)
            outputs = model(img_embeddings, text_embeddings)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = total_loss / len(data_loader)
    val_acc = correct / total
    return val_loss, val_acc

train_model()
wandb.finish()
