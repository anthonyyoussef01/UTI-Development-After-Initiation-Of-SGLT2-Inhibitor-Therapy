import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings

warnings.filterwarnings('ignore')

# Set device and optimize CUDA performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Dataset class for UTI prediction
class UTIDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Data loading and preprocessing function
def load_and_preprocess_data(file_path='sglt2_uti_data.csv'):
    # Load data exported from SQL query
    df = pd.read_csv(file_path)

    # Create binary target: UTI (1) vs No UTI (0)
    df['uti_target'] = df['uti_status'].apply(
        lambda x: 1 if x in ['Recurrent UTI', 'New onset UTI'] else 0
    )

    # Encode categorical variables
    categorical_cols = ['sglt2_drug', 'gender', 'admission_type']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Convert boolean columns to integers
    bool_cols = ['has_diabetes', 'uti_history']
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Select features for modeling
    feature_cols = [
        'sglt2_drug_encoded', 'sglt2_duration', 'has_diabetes', 'uti_history',
        'gender_encoded', 'age_at_admission', 'admission_type_encoded',
        'length_of_stay'
    ]

    # Handle missing values
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())

    # Scale numerical features
    numerical_cols = ['sglt2_duration', 'age_at_admission', 'length_of_stay']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Prepare features and target
    X = df[feature_cols].values
    y = df['uti_target'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols


# Transformer Model for UTI prediction
# Transformer Model for UTI prediction
class UTITransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super(UTITransformer, self).__init__()

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
            # Removed the attn_implementation parameter
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output layers remain unchanged
        self.output_layer1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer2 = nn.Linear(hidden_dim // 2, 2)  # Binary classification

    # The forward method remains unchanged
    def forward(self, x):
        # Input embedding with sequence dimension
        x = self.input_embedding(x).unsqueeze(1)

        # Add positional encoding
        x = x + self.pos_encoder

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Output layers
        x = x.squeeze(1)
        x = self.output_layer1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.output_layer2(x)

        return logits


# Mixed precision training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    # Setup gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    for batch in dataloader:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        # Mixed precision training
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(features)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )

    return total_loss / len(dataloader), accuracy, precision, recall, f1


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision evaluation
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs[:, 1].cpu().numpy())
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )

    return total_loss / len(dataloader), accuracy, precision, recall, f1, predictions, true_labels, probabilities


# Standard training function (replacing PBT version)
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, input_dim, config):
    # Create model with config hyperparameters
    model = UTITransformer(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    # Create datasets and dataloaders
    train_dataset = UTIDataset(X_train, y_train)
    val_dataset = UTIDataset(X_val, y_val)
    test_dataset = UTIDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Loss function and optimizer
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"]
    )

    # Training loop
    best_val_f1 = 0
    best_model_state = model.state_dict().copy()

    print("\nTraining started...")
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Step the scheduler
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch + 1}/{config['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Save best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation F1: {val_f1:.4f}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    # Evaluate on test set
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    print("\nTest Set Evaluation:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=["No UTI", "UTI"]))

    # Save the best model
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'input_dim': input_dim
    }, "sglt2_uti_best_model.pt")
    print("Best model saved to 'sglt2_uti_best_model.pt'")

    return model, test_acc, test_f1


# Main function to run the entire pipeline
def main():
    print("Starting UTI prediction with SGLT2 data...")

    # Load and preprocess data from exported SQL query
    X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess_data()

    # Split train into train and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # Set hyperparameters manually
    config = {
        "batch_size": 16,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "hidden_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "num_epochs": 30
    }

    print("Model configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train model with fixed hyperparameters
    model, test_acc, test_f1 = train_model(
        X_train_final, y_train_final,
        X_val, y_val,
        X_test, y_test,
        input_dim,
        config
    )

    print(f"\nTraining completed! Final test accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")


if __name__ == "__main__":
    main()
