"""
Module for Named Entity Recognition (NER) training using BERT

This module contains classes and functions for loading datasets, 
initializing the model, training the model, and implementing early stopping.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

# Global variables
MAX_LEN = 128
BATCH_SIZE = 4
NUM_EPOCHS = 5
PATIENCE = 3
LR = 2e-5
WARMUP_STEPS_RATIO = 0.1


class NERDataset(Dataset):
    """
    Custom dataset class for NER

    Args:
        texts (list): List of text samples
        tags (list): List of corresponding tags for each text sample
        tokenizer (BertTokenizer): BERT tokenizer
        max_len (int): Maximum sequence length
        tag2id (dict): Mapping from tags to IDs

    Methods:
        __getitem__(idx): Gets a single data sample at the given index
        __len__(): Returns the length of the dataset
    """

    def __init__(self, texts, tags, tokenizer, max_len, tag2id):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id

    def __getitem__(self, idx):
        """
        Gets a single data sample

        Args:
            idx (int): Index of the sample

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        text = self.texts[idx]
        tags = self.tags[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        labels = torch.tensor(tags, dtype=torch.long)

        # Ensure labels length matches input length
        if len(labels) < self.max_len:
            labels = torch.nn.functional.pad(labels, (0, self.max_len - len(labels)), 'constant', 0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __len__(self):
        """
        Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.texts)


def load_dataset(path, tag2id):
    """
    Loads the dataset from a JSON file

    Args:
        path (str): Path to the JSON file
        tag2id (dict): Mapping from tags to IDs

    Returns:
        tuple: Lists of texts and tags
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    tags = [[tag2id[tag] for tag in item["tags"]] for item in data]  # Directly convert tags to IDs
    return texts, tags


class EarlyStopping:
    """
    Early stopping class to monitor validation loss and stop training if no improvement

    Args:
        patience (int): Number of epochs to wait for improvement before stopping
        verbose (bool): Whether to print verbose messages
        delta (float): Minimum change in the monitored quantity to qualify as an improvement
        path (str): Path to save the best model's state_dict

    Methods:
        __call__(val_loss, model): Called in each epoch to check for early stopping
    """

    def __init__(self, patience, verbose=False, delta=0, path='best_point.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """
        Checks for early stopping based on the validation loss

        Args:
            val_loss (float): Current validation loss
            model (torch.nn.Module): Model to save if there's an improvement

        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f'EarlyStopping: best model saved to {self.path}')


# Load tag to ID mapping
tag_filepath = "data/tag2id.json"
tag2id = json.load(open(tag_filepath, 'r', encoding='utf-8'))

# Load datasets
train_texts, train_tags = load_dataset('data/train.json', tag2id)
eval_texts, eval_tags = load_dataset('data/eval.json', tag2id)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset instances
train_dataset = NERDataset(train_texts, train_tags, tokenizer, MAX_LEN, tag2id)
eval_dataset = NERDataset(eval_texts, eval_tags, tokenizer, MAX_LEN, tag2id)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize BERT model for token classification
model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                   num_labels=len(tag2id))

# Configure optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=WARMUP_STEPS_RATIO * num_training_steps,
                                            num_training_steps=num_training_steps)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize early stopping object
early_stopping = EarlyStopping(PATIENCE, verbose=True)

# Training loop
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Update learning rate
        scheduler.step()

        total_loss += loss.item()
        batch_loss = total_loss / (step + 1)

        if (step + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Batch {step + 1}, Loss: {batch_loss}')

    # Evaluation at the end of each epoch
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()

    print(f'Epoch {epoch + 1}, Validation Loss: {eval_loss / len(eval_dataloader)}')

    # Early stopping check
    if early_stopping(eval_loss / len(eval_dataloader), model):
        print("Early stopping")
        break

# Load best model weights if early stopping triggered
if early_stopping.early_stop:
    model.load_state_dict(torch.load(early_stopping.path))
    print(f'Loaded best model weights from {early_stopping.path}')

# Save the model
model.save_pretrained('./my_bert_ner_model')
