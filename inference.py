import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model_path = "models/distilbert-base-uncased-ag_news"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax(dim=1).item()
    label = model.config.id2label[predicted_class_id]
    return label

print(predict("Boris Becker wins Wimbledon."))