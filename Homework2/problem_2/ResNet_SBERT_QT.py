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

'''
train_counts = train_df['question_type'].value_counts()
val_counts = val_df['question_type'].value_counts()
test_counts = test_df['question_type'].value_counts()

# Combine into a summary table
summary_df = pd.DataFrame({
    'Training Set': train_counts,
    'Validation Set': val_counts,
    'Test Set': test_counts
}).fillna(0).astype(int)

# Display the summary table
print("\nSummary of Question Types Across Datasets:\n", summary_df)
'''

# Create label maps for each question type
label_maps = {
    q_type: {answer: idx for idx, answer in enumerate(train_df[train_df['question_type'] == q_type]['answer'].unique())}
    for q_type in train_df['question_type'].unique()
}

# Log label map size and print label map for each type
for q_type, label_map in label_maps.items():
    wandb.config.update({f"{q_type}_label_map_size": len(label_map)})
    print(f"Label Map for {q_type}: {label_map}")

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

        # Check if img_embedding is 1D after squeeze, convert it to 2D by adding a batch dimension
        if len(img_embedding.shape) == 1:
            img_embedding = img_embedding.unsqueeze(0)

        img_embedding = nn.ReLU()(self.fc_img(img_embedding))  # Apply Linear + ReLU
        return img_embedding

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element is token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Textual Encoder using HuggingFace Transformers
class TextualEncoder(nn.Module):
    def __init__(self):
        super(TextualEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.fc_text = nn.Linear(384, 128)  # Linear layer to reduce dimensions to 128

    def forward(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        text_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        text_embedding = F.normalize(text_embedding, p=2, dim=1)  # Normalize embeddings
        text_embedding = nn.ReLU()(self.fc_text(text_embedding))  # Apply Linear + ReLU

        # Check if text_embedding is 1D after reduction, convert it to 2D by adding a batch dimension
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.unsqueeze(0)

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
        # Ensure that both embeddings have the same number of dimensions
        if len(img_emb.shape) == 1:
            img_emb = img_emb.unsqueeze(0)
        if len(text_emb.shape) == 1:
            text_emb = text_emb.unsqueeze(0)

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

# Define `train_model` to handle individual question types
def train_model(train_loader, val_loader, question_type, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = VisualEncoder().to(device)
    textual_encoder = TextualEncoder().to(device)
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
