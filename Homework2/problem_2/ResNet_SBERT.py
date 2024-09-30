import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import wandb

# Initialize W&B
wandb.init(project="vqa-resnet-sbert", config={
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4, # Avoid Too Small Init.
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

# Create label map from unique answers
all_answers = pd.concat([train_df['answer'], val_df['answer'], test_df['answer']])
unique_answers = all_answers.unique()
label_map = {answer: idx for idx, answer in enumerate(unique_answers)}

# Log label map size and print label map
wandb.config.update({"label_map_size": len(label_map)})
print(f"Label Map: {label_map}")

# Create Datasets and DataLoaders
train_dataset = CustomVQADataset(dataframe=train_df, label_map=label_map, transform=image_transform)
val_dataset = CustomVQADataset(dataframe=val_df, label_map=label_map, transform=image_transform)
test_dataset = CustomVQADataset(dataframe=test_df, label_map=label_map, transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# Visual Encoder using ResNet-50
class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove classification head
        self.fc_img = nn.Linear(2048, 128)  # Linear layer to reduce dimensions to 128

    def forward(self, x):
        with torch.no_grad():
            img_embedding = self.resnet(x).squeeze()
        img_embedding = nn.ReLU()(self.fc_img(img_embedding))  # Apply Linear + ReLU
        return img_embedding

# Define mean pooling function for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element is token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Textual Encoder using HuggingFace Transformers (Updated)
class TextualEncoder(nn.Module):
    def __init__(self):
        super(TextualEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.fc_text = nn.Linear(384, 128)  # Linear layer to reduce dimensions to 128

    def forward(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # **Move inputs to the same device as the model**
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        # Compute embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Apply mean pooling
        text_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        text_embedding = F.normalize(text_embedding, p=2, dim=1)  # Normalize embeddings
        text_embedding = nn.ReLU()(self.fc_text(text_embedding))  # Apply Linear + ReLU
        return text_embedding

# Combined Model for Multi-Modal Fusion
class CombinedModel(nn.Module):
    def __init__(self, reduced_dim=256, hidden_dim=512, num_classes=30):
        super(CombinedModel, self).__init__()
        self.bn = nn.BatchNorm1d(reduced_dim)  # BatchNorm layer for combined embeddings
        self.fc1 = nn.Sequential(
            nn.Linear(reduced_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, img_emb, text_emb):

        # Concatenate image and text embeddings
        concatenated_emb = torch.cat((img_emb, text_emb), dim=-1)  # 128 + 128 = 256
        norm_emb = self.bn(concatenated_emb)  # Normalize the combined embeddings
        hidden_emb = self.fc1(norm_emb)
        output = self.classifier(hidden_emb)
        wandb.log({
            "Image Embedding Dimension": str(img_emb.shape),
            "Text Embedding Dimension": str(text_emb.shape),
            "Concatenated Embedding Dimension": str(concatenated_emb.shape)
        })
        print(f"Image Embedding Dimension: {img_emb.shape}")
        print(f"Text Embedding Dimension: {text_emb.shape}")
        print(f"Concatenated Embedding Dimension: {concatenated_emb.shape}")
        return output

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = VisualEncoder().to(device)
    textual_encoder = TextualEncoder().to(device)
    combined_model = CombinedModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(combined_model.parameters(), lr=wandb.config.learning_rate)

    for epoch in range(wandb.config.epochs):
        combined_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (questions, images, labels) in enumerate(train_loader):
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
            wandb.log({"step_loss": loss.item()})

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # Log epoch-wise training metrics
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

    # Final evaluation on test set with updated evaluate function
    test_loss, test_acc = evaluate(combined_model, test_loader, visual_encoder, textual_encoder, device)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

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
