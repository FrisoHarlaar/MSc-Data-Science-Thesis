import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW  

import pandas as pd
import numpy as np
import os
import os.path as osp
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import logging

from artemis.emotions import ARTEMIS_EMOTIONS, IDX_TO_EMOTION, positive_negative_else
from artemis.in_out.basics import create_dir

# Configuration
num_labels = len(ARTEMIS_EMOTIONS)
model_name = 'google-bert/bert-base-uncased'
load_best_model = False
do_training = False
max_train_epochs = 50
subsample_data = False

# Paths - update these to your paths
my_out_dir = r'..\predictions\bert_based'
preprocessed_artemis = r'..\preprocessed_data\artemis_preprocessed.csv'
create_dir(my_out_dir)
best_model_dir = osp.join(my_out_dir, 'best_model')
create_dir(best_model_dir)

# Training parameters
batch_size = 16  # Reduced from 128 due to memory constraints
max_length = 512
learning_rate = 2e-5
warmup_steps = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArtemisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load and prepare the Artemis dataset"""
    df = pd.read_csv(preprocessed_artemis)
    
    data_splits = {}
    for split in ['train', 'test', 'val']:
        mask = (df['split'] == split)
        sub_df = df[mask][['utterance_spelled', 'emotion_label']].copy()
        sub_df.reset_index(drop=True, inplace=True)
        sub_df.columns = ["text", "labels"]
        
        if subsample_data:
            sub_df = sub_df.sample(1000)
            sub_df.reset_index(drop=True, inplace=True)
        
        data_splits[split] = sub_df
    
    return data_splits

def create_data_loaders(data_splits, tokenizer):
    """Create PyTorch DataLoaders"""
    datasets = {}
    data_loaders = {}
    
    for split_name, split_data in data_splits.items():
        dataset = ArtemisDataset(
            split_data['text'], 
            split_data['labels'], 
            tokenizer, 
            max_length
        )
        datasets[split_name] = dataset
        
        shuffle = (split_name == 'train')
        data_loaders[split_name] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=4
        )
    
    return data_loaders, datasets

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        accuracy = correct_predictions / total_predictions
        
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'accuracy': accuracy
        })
    
    return total_loss / len(data_loader), accuracy

def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    print("Loading data...")
    data_splits = load_data()
    
    # Initialize tokenizer and model
    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if load_best_model and os.path.exists(best_model_dir):
        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    
    model.to(device)
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders, datasets = create_data_loaders(data_splits, tokenizer)
    
    if do_training:
        print("Starting training...")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(data_loaders['train']) * max_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        
        for epoch in range(max_train_epochs):
            print(f"\nEpoch {epoch + 1}/{max_train_epochs}")
            
            # Training
            train_loss, train_accuracy = train_epoch(
                model, data_loaders['train'], optimizer, scheduler, device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            # Validation
            val_loss, val_accuracy, _, _ = evaluate(model, data_loaders['val'], device)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
    
    # Load best model for evaluation
    print("Loading best model for final evaluation...")
    if os.path.exists(best_model_dir):
        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
        model.to(device)
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy, predictions, labels = evaluate(
        model, data_loaders['test'], device
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Positive/Negative emotion analysis
    print("\nAnalyzing positive/negative emotions...")
    
    gt = pd.Series(labels)
    predictions_series = pd.Series(predictions)
    
    # Convert to positive/negative/else
    gt_pne = gt.apply(lambda x: positive_negative_else(IDX_TO_EMOTION[x]))
    predictions_pne = predictions_series.apply(lambda x: positive_negative_else(IDX_TO_EMOTION[x]))
    
    ternary_accuracy = (gt_pne == predictions_pne).mean()
    print(f'Ternary prediction accuracy (pos/neg/else): {ternary_accuracy:.4f}')
    
    # Binary accuracy (dropping "something else")
    se_label = positive_negative_else('something else')
    gt_pn = gt_pne[gt_pne != se_label]
    gt_pn.reset_index(drop=True, inplace=True)
    
    pred_pn = predictions_pne[(gt_pne != se_label).values]
    pred_pn.reset_index(drop=True, inplace=True)
    
    binary_accuracy = (gt_pn == pred_pn).mean()
    print(f'Binary prediction accuracy (pos/neg only): {binary_accuracy:.4f}')
    
    # Save results
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'ternary_accuracy': ternary_accuracy,
        'binary_accuracy': binary_accuracy
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(osp.join(my_out_dir, 'test_results.csv'), index=False)
    
    print(f"\nResults saved to {my_out_dir}")

if __name__ == "__main__":
    main()