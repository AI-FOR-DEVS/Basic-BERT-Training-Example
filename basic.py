import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.optim import AdamW

params = {
    'model_name': 'distilbert-base-cased',
    'learning_rate': 5e-5, #0.00005
    'batch_size': 16,
    'num_epochs': 1,
    'dataset_name': 'ag_news',
    'task_name': 'sequence_classification',
    'log_steps': 100,
    'max_seq_length': 128,
    'output_dir': 'models/distilbert-base-uncased-ag_news',
}

dataset = load_dataset(params['dataset_name'])
tokenizer = DistilBertTokenizer.from_pretrained(params['model_name'])

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=params['max_seq_length'])

train_dataset = dataset["train"].shuffle().select(range(5_000)).map(tokenize, batched=True)
test_dataset = dataset["test"].shuffle().select(range(100)).map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

labels = dataset["train"].features['label'].names

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained(params['model_name'], num_labels=len(labels))

model.config.id2label = {i: label for i, label in enumerate(labels)}
params['id2label'] = model.config.id2label

optimizer = AdamW(model.parameters(), lr=params['learning_rate'])

# Just 1 epoch

for i, batch in enumerate(train_loader, 0):
  inputs, masks, labels = batch['input_ids'], batch['attention_mask'], batch['label']
  optimizer.zero_grad()
  outputs = model(inputs, attention_mask=masks, labels=labels)
  loss = outputs.loss
  loss.backward() # compute gradients (backpropagation)
  optimizer.step() # update weights using gradients
  if i % params['log_steps'] == 0:
    print(f"Step {i}, Loss: {loss.item():.4f}")
  
os.makedirs(params['output_dir'], exist_ok=True)
model.save_pretrained(params['output_dir'])
tokenizer.save_pretrained(params['output_dir'])