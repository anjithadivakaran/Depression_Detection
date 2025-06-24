
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import multimodal_dataset
from models import depression_model
from utils import config , metrics



def train_model(json_path):
    dataset = multimodal_dataset.MultimodelDataset(json_path)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)
    test_loader = DataLoader(test_set, batch_size=4)

    model = depression_model.DepressionClassifier().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    
    # === Training ===
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0
        for input_ids, attn_mask, images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids, attn_mask, images, labels = input_ids.to(config.DEVICE), attn_mask.to(config.DEVICE), images.to(config.DEVICE), labels.to(config.DEVICE)
            logits = model(input_ids, attn_mask, images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Train Loss: {total_loss/len(train_loader):.4f}")

        # === Validation ===
        model.eval()
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for input_ids, attn_mask, images, labels in val_loader:
                input_ids, attn_mask, images = input_ids.to(config.DEVICE), attn_mask.to(config.DEVICE), images.to(config.DEVICE)
                logits = model(input_ids, attn_mask, images)
                all_val_preds.extend(logits.cpu())
                all_val_labels.extend(labels)
        val_acc, val_f1 = metrics.compute_metrics(torch.stack(all_val_preds), torch.tensor(all_val_labels))
        print(f"Epoch {epoch} Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

    # === Test ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, images, labels in test_loader:
            input_ids, attn_mask, images = input_ids.to(config.DEVICE), attn_mask.to(config.DEVICE), images.to(config.DEVICE)
            logits = model(input_ids, attn_mask, images)
            all_preds.extend(logits.cpu())
            all_labels.extend(labels)

    test_acc, test_f1 = metrics.compute_metrics(torch.stack(all_preds), torch.tensor(all_labels))
    print(f"\nFinal Test Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")


# === Run ===
if __name__ == "__main__":
    train_model("processed_user_data.json")
