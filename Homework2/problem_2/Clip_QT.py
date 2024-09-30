import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
import pandas as pd
import os
import clip
from torch.utils.data import Dataset, DataLoader
import wandb

# Initialize W&B
wandb.init(project="vqa-clip", config={
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,  # Avoid Too Small Init.
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

        # Ensure that the answer is correctly mapped
        if answer in self.label_map:
            label = self.label_map[answer]
        else:
            return None

        # Construct image path and handle missing images
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            return None

        # Load and transform the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return question, image, label

# Define image transformations for ResNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CSV files for training, validation, and test sets
train_df = pd.read_csv('data/new_data_train.csv')
val_df = pd.read_csv('data/new_data_val.csv')
test_df = pd.read_csv('data/new_data_test.csv')

# Define function to classify question types based on answer patterns
def classify_by_answer(answer):
    # Check if the answer is a count (i.e., a numeric string)
    if str(answer).isdigit():
        return 'Count-Based'
    # Check if the answer is a color (we'll use a predefined set of common colors)
    colors = {'red', 'blue', 'green', 'yellow', 'white', 'black', 'brown', 'gray', 'pink', 'orange'}
    if str(answer).lower() in colors:
        return 'Attribute-Based'
    # Otherwise, it's an object identification or relational question
    return 'Object Identification/Relational'

# Apply the classification function to categorize questions in each dataset
train_df['question_type'] = train_df['answer'].apply(classify_by_answer)
val_df['question_type'] = val_df['answer'].apply(classify_by_answer)
test_df['question_type'] = test_df['answer'].apply(classify_by_answer)

# Create label maps for each question type
label_maps = {
    q_type: {answer: idx for idx, answer in enumerate(train_df[train_df['question_type'] == q_type]['answer'].unique())}
    for q_type in train_df['question_type'].unique()
}

# Log label map size and print label map for each type
for q_type, label_map in label_maps.items():
    wandb.config.update({f"{q_type}_label_map_size": len(label_map)})
    print(f"Label Map for {q_type}: {label_map}")

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

def train_model(train_loader, val_loader, question_type, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = VisualEncoder(device).to(device)
    textual_encoder = TextualEncoder(device=device).to(device)
    combined_model = CombinedModel(num_classes=num_classes).to(device)

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

            # Log step-wise loss to W&B
            wandb.log({f"{question_type}_step_loss": loss.item()})

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # Log epoch-wise training metrics for this question type
        wandb.log({f"{question_type}_train_loss": train_loss, f"{question_type}_train_accuracy": train_acc})

    # Final evaluation on validation set for this question type
    val_loss, val_acc = evaluate(combined_model, val_loader, visual_encoder, textual_encoder, device, question_type)
    wandb.log({f"{question_type}_val_loss": val_loss, f"{question_type}_val_accuracy": val_acc})
    print(f"Final Validation Loss ({question_type}): {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Update `evaluate` function to use question type for better tracking in W&B
def evaluate(model, data_loader, visual_encoder, textual_encoder, device, question_type):
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

# =============================
#  Main Loop for Each Question Type
# =============================

# Loop through each question type and create separate train and validation loaders
for question_type, label_map in label_maps.items():
    print(f"Training and evaluating for question type: {question_type}")

    # Filter datasets based on the current question type
    train_subset = train_df[train_df['question_type'] == question_type]
    val_subset = val_df[val_df['question_type'] == question_type]

    # Create custom datasets for each subset
    train_dataset = CustomVQADataset(dataframe=train_subset, label_map=label_map, transform=image_transform)
    val_dataset = CustomVQADataset(dataframe=val_subset, label_map=label_map, transform=image_transform)

    # Create data loaders for this question type
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    # Get number of classes for this question type
    num_classes = len(label_map)

    # Call the train_model function for this question type
    train_model(train_loader, val_loader, question_type, num_classes)

wandb.finish()
