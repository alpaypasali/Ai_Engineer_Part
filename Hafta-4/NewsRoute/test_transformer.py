import json
import torch
from transformer_model import NewsRouteTransformer


with open("models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)


model = NewsRouteTransformer(vocab_size=len(vocab))

model.load_state_dict(
    torch.load(
        "models/newsroute_transformer.pt",
        map_location="cpu",
        weights_only=True
    )
)


model.eval()

print("Vocabulary Size:", len(vocab))
print("Transformer başarıyla yüklendi.")
print("Model Device:", next(model.parameters()).device)


text = "NASA launches a new spacecraft to explore Mars"

tokens = text.lower().split()


token_ids = [vocab.get(token, 1) for token in tokens]


token_ids = token_ids[:64]
token_ids += [0] * (64 - len(token_ids))


input_ids = torch.tensor([token_ids], dtype=torch.long)

with torch.no_grad():
    logits = model(input_ids)
    prediction = torch.argmax(logits, dim=1).item()

class_names = ["World", "Sports", "Business", "Sci/Tech"]

print("Prediction ID:", prediction)
print("Category:", class_names[prediction])
print("Logits Shape:", logits.shape)


import pandas as pd

val_df = pd.read_csv("data/val_split.csv")

text = val_df.iloc[0]["text"]

tokens = text.lower().split()
token_ids = [vocab.get(token, 1) for token in tokens]

token_ids = token_ids[:64]
token_ids += [0] * (64 - len(token_ids))

input_ids = torch.tensor([token_ids], dtype=torch.long)

with torch.no_grad():
    logits = model(input_ids)
    prediction = torch.argmax(logits, dim=1).item()

print("Gerçek etiket:", val_df.iloc[0]["label"])
print("Transformer tahmini:", prediction)