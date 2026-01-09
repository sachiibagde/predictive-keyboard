import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from collections import Counter
import random

# -----------------------
# Download tokenizer
# -----------------------
nltk.download('punkt')

# -----------------------
# Load dataset
# -----------------------
file_path = r"C:\Users\LENOVO\OneDrive\Documents\Desktop\ML Projects\sherlock-holm.es_stories_plain-text_advs.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = word_tokenize(text)
print("Total Tokens:", len(tokens))

# -----------------------
# Build vocabulary
# -----------------------
word_counts = Counter(tokens)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

print("Vocabulary Size:", vocab_size)

# -----------------------
# Create sequences
# -----------------------
sequence_length = 4  # 3 words → predict next

data = []
for i in range(len(tokens) - sequence_length):
    input_seq = tokens[i:i + sequence_length - 1]
    target = tokens[i + sequence_length - 1]
    data.append((input_seq, target))

def encode(seq):
    return [word2idx[word] for word in seq]

encoded_data = [
    (torch.tensor(encode(inp)), torch.tensor(word2idx[target]))
    for inp, target in data
]

# -----------------------
# Model
# -----------------------
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = PredictiveKeyboard(vocab_size)

# -----------------------
# Training setup
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# -----------------------
# Training loop
# -----------------------
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    random.shuffle(encoded_data)

    for input_seq, target in encoded_data[:10000]:  # limit for speed
        input_seq = input_seq.unsqueeze(0)

        output = model(input_seq)
        loss = criterion(output, target.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f}")

# -----------------------
# Prediction function
# -----------------------
def suggest_next_words(model, text_prompt, top_k=3):
    model.eval()
    tokens = word_tokenize(text_prompt.lower())

    if len(tokens) < sequence_length - 1:
        raise ValueError("Enter at least 3 words")

    input_seq = tokens[-(sequence_length - 1):]
    input_tensor = torch.tensor(encode(input_seq)).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, top_k).indices.tolist()

    return [idx2word[idx] for idx in top_indices]

# -----------------------
# Test
# -----------------------
print("\nSuggestions:", suggest_next_words(model, "So, are we really at"))
