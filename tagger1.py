import torch
import torch.nn as nn
import torch.optim as optim
import itertools

# Load data functions
def load_data(file_path):
    sentences = []
    tags = []
    with open(file_path, 'r') as file:
        sentence = []
        tag_seq = []
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence = []
                    tag_seq = []
            else:
                word, tag = line.split()
                sentence.append(word)
                tag_seq.append(tag)
        if sentence:
            sentences.append(sentence)
            tags.append(tag_seq)
    
    print(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences, tags

def load_test_data(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        sentence = []
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word = line
                sentence.append(word)
        if sentence:
            sentences.append(sentence)
    
    print(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences

# Example usage
pos_train_sentences, pos_train_tags = load_data('pos/train')
pos_dev_sentences, pos_dev_tags = load_data('pos/dev')
pos_test_sentences = load_test_data('pos/test')
ner_train_sentences, ner_train_tags = load_data('ner/train')
ner_dev_sentences, ner_dev_tags = load_data('ner/dev')
ner_test_sentences = load_test_data('ner/test')

# Vocabulary and tag set functions
def build_vocab(sentences):
    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    index_to_word = {0: '<PAD>', 1: '<UNK>'}
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_index:
                idx = len(word_to_index)
                word_to_index[word] = idx
                index_to_word[idx] = word
    return word_to_index, index_to_word

def build_tagset(tags):
    tag_to_index = {}
    index_to_tag = {}
    for tag_seq in tags:
        for tag in tag_seq:
            if tag not in tag_to_index:
                idx = len(tag_to_index)
                tag_to_index[tag] = idx
                index_to_tag[idx] = tag
    return tag_to_index, index_to_tag

# Build vocabulary and tag sets for POS and NER
word_to_index, index_to_word = build_vocab(pos_train_sentences + ner_train_sentences)
print("Vocabulary size:", len(word_to_index))

pos_tag_to_index, pos_index_to_tag = build_tagset(pos_train_tags)
ner_tag_to_index, ner_index_to_tag = build_tagset(ner_train_tags)
print("POS Tag set size:", len(pos_tag_to_index))
print("NER Tag set size:", len(ner_tag_to_index))

# Model definition
class WindowBasedTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size, hidden_dim, output_dim):
        super(WindowBasedTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(window_size * embedding_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeds = self.embedding(x).view((x.size(0), -1))
        hidden = self.tanh(self.fc1(embeds))
        output = self.fc2(hidden)
        return output

# Data preparation function
def prepare_data(sentences, tags, word_to_index, tag_to_index, window_size):
    X = []
    y = []
    half_window = window_size // 2
    for sentence, tag_seq in zip(sentences, tags):
        sentence_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
        tag_indices = [tag_to_index[tag] for tag in tag_seq]
        padded_sentence = [word_to_index['<PAD>']] * half_window + sentence_indices + [word_to_index['<PAD>']] * half_window
        for i in range(len(sentence)):
            window = padded_sentence[i:i + window_size]
            X.append(window)
            y.append(tag_indices[i])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def prepare_test_data(sentences, word_to_index, window_size):
    X = []
    half_window = window_size // 2
    for sentence in sentences:
        sentence_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
        padded_sentence = [word_to_index['<PAD>']] * half_window + sentence_indices + [word_to_index['<PAD>']] * half_window
        for i in range(len(sentence)):
            window = padded_sentence[i:i + window_size]
            X.append(window)
    return torch.tensor(X, dtype=torch.long)

# Parameters
window_size = 5
embedding_dim = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare training, dev, and test data for POS and NER
X_train_pos, y_train_pos = prepare_data(pos_train_sentences, pos_train_tags, word_to_index, pos_tag_to_index, window_size)
X_dev_pos, y_dev_pos = prepare_data(pos_dev_sentences, pos_dev_tags, word_to_index, pos_tag_to_index, window_size)
X_test_pos = prepare_test_data(pos_test_sentences, word_to_index, window_size)
X_train_pos, y_train_pos = X_train_pos.to(device), y_train_pos.to(device)
X_dev_pos, y_dev_pos = X_dev_pos.to(device), y_dev_pos.to(device)
X_test_pos = X_test_pos.to(device)

X_train_ner, y_train_ner = prepare_data(ner_train_sentences, ner_train_tags, word_to_index, ner_tag_to_index, window_size)
X_dev_ner, y_dev_ner = prepare_data(ner_dev_sentences, ner_dev_tags, word_to_index, ner_tag_to_index, window_size)
X_test_ner = prepare_test_data(ner_test_sentences, word_to_index, window_size)
X_train_ner, y_train_ner = X_train_ner.to(device), y_train_ner.to(device)
X_dev_ner, y_dev_ner = X_dev_ner.to(device), y_dev_ner.to(device)
X_test_ner = X_test_ner.to(device)

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total * 100
    return accuracy

import matplotlib.pyplot as plt

# Function to train model and collect metrics
def train_model_collect_metrics(X_train, y_train, X_dev, y_dev, vocab_size, output_dim, learning_rate, hidden_dim, num_epochs):
    model = WindowBasedTagger(vocab_size, embedding_dim, window_size, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    dev_losses = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        train_accuracy = calculate_accuracy(train_outputs, y_train)
        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy)
        
        model.eval()
        with torch.no_grad():
            dev_outputs = model(X_dev)
            dev_loss = criterion(dev_outputs, y_dev)
            dev_accuracy = calculate_accuracy(dev_outputs, y_dev)
            dev_losses.append(dev_loss.item())
            dev_accuracies.append(dev_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Dev Loss: {dev_loss.item():.4f}, Dev Accuracy: {dev_accuracy:.2f}%')

    return model, train_losses, train_accuracies, dev_losses, dev_accuracies

# Function to save predictions
def save_predictions(model, X_test, index_to_tag, output_file, test_sentences):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
    
    with open(output_file, 'w') as f:
        pred_idx = 0
        for sentence in test_sentences:
            for word in sentence:
                pred_tag = index_to_tag[predicted[pred_idx].item()]
                f.write(f"{word} {pred_tag}\n")
                pred_idx += 1
            f.write("\n")  # Blank line to indicate sentence boundary
    print(f"Predictions saved to {output_file}")

# Best hyperparameters
best_pos_params = {'learning_rate': 0.005, 'hidden_dim': 50}
# Number of epochs
num_epochs = 200
# Train POS model and collect metrics
print("Training POS model with best hyperparameters...")
pos_model, pos_train_losses, pos_train_accuracies, pos_dev_losses, pos_dev_accuracies = train_model_collect_metrics(
    X_train_pos, y_train_pos, X_dev_pos, y_dev_pos, len(word_to_index), len(pos_tag_to_index),
    best_pos_params['learning_rate'], best_pos_params['hidden_dim'], num_epochs
)

# Plotting function
def plot_metrics(train_losses, train_accuracies, dev_losses, dev_accuracies, title):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, dev_losses, label='Dev Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, dev_accuracies, label='Dev Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot metrics for POS model
plot_metrics(pos_train_losses, pos_train_accuracies, pos_dev_losses, pos_dev_accuracies, 'POS Model')

# Save POS test predictions
save_predictions(pos_model, X_test_pos, pos_index_to_tag, 'test1.pos', pos_test_sentences)

# Train NER model and collect metrics
best_ner_params = {'learning_rate': 0.003, 'hidden_dim': 10}
# Number of epochs
num_epochs = 150
print("Training NER model with best hyperparameters...")
ner_model, ner_train_losses, ner_train_accuracies, ner_dev_losses, ner_dev_accuracies = train_model_collect_metrics(
    X_train_ner, y_train_ner, X_dev_ner, y_dev_ner, len(word_to_index), len(ner_tag_to_index),
    best_ner_params['learning_rate'], best_ner_params['hidden_dim'], num_epochs
)

# Plot metrics for NER model
plot_metrics(ner_train_losses, ner_train_accuracies, ner_dev_losses, ner_dev_accuracies, 'NER Model')

# Save NER test predictions
save_predictions(ner_model, X_test_ner, ner_index_to_tag, 'test1.ner', ner_test_sentences)
