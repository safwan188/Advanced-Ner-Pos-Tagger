import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
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
class TaggingDataset(Dataset):
    def __init__(self, X_words, X_chars, y):
        self.X_words = X_words
        self.X_chars = X_chars
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_words[idx], self.X_chars[idx], self.y[idx]
# Load pre-trained embeddings and vocabulary
vectors = np.loadtxt("wordVectors.txt")
with open("vocab.txt", "r") as f:
    vocab = [line.strip() for line in f]

# Create a dictionary mapping words to their vectors
word_to_vec = {word: vector for word, vector in zip(vocab, vectors)}

# Vocabulary and tag set functions
def build_vocab(sentences, pre_trained_vocab):
    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    index_to_word = {0: '<PAD>', 1: '<UNK>'}
    for word in pre_trained_vocab:
        if word not in word_to_index:
            idx = len(word_to_index)
            word_to_index[word] = idx
            index_to_word[idx] = word
    return word_to_index, index_to_word

def build_char_vocab(sentences):
    char_to_index = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            for char in word:
                if char not in char_to_index:
                    char_to_index[char] = len(char_to_index)
    return char_to_index

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

# Model definition
class CharacterLevelCNN(nn.Module):
    def __init__(self, num_chars, char_embedding_dim, num_filters, kernel_size):
        super(CharacterLevelCNN, self).__init__()
        self.char_embedding = nn.Embedding(num_chars, char_embedding_dim)
        self.conv = nn.Conv1d(char_embedding_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(0.5)  # As mentioned in the paper

    def forward(self, x):
        # x shape: (batch_size, max_word_len)
        x = self.dropout(self.char_embedding(x))
        x = x.transpose(1, 2)  # (batch_size, char_embedding_dim, max_word_len)
        x = self.conv(x)
        x = F.relu(x)
        x, _ = torch.max(x, dim=2)  # Max-pooling
        return x

class WindowBasedTaggerWithCNN(nn.Module):
    def __init__(self, vocab_size, num_chars, char_embedding_dim, num_filters, kernel_size, word_embedding_dim, window_size, hidden_dim, output_dim, pre_trained_embeddings=None):
        super(WindowBasedTaggerWithCNN, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        if pre_trained_embeddings is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pre_trained_embeddings))
        self.char_cnn = CharacterLevelCNN(num_chars, char_embedding_dim, num_filters, kernel_size)
        self.fc1 = nn.Linear(window_size * (word_embedding_dim + num_filters), hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, words, chars):
        # words shape: (batch_size, window_size)
        # chars shape: (batch_size, window_size, max_word_len)
        word_embeds = self.word_embedding(words)
        char_embeds = self.char_cnn(chars.view(-1, chars.size(2))).view(chars.size(0), chars.size(1), -1)
        combined_embeds = torch.cat([word_embeds, char_embeds], dim=2)
        embeds = combined_embeds.view((words.size(0), -1))
        hidden = self.tanh(self.fc1(embeds))
        output = self.fc2(hidden)
        return output

# Data preparation function
def prepare_data_with_chars(sentences, tags, word_to_index, char_to_index, tag_to_index, window_size, max_word_len):
    X_words, X_chars, y = [], [], []
    half_window = window_size // 2
    for sentence, tag_seq in zip(sentences, tags):
        word_indices = [word_to_index.get(word.lower(), word_to_index['<UNK>']) for word in sentence]
        char_indices = [[char_to_index.get(char, char_to_index['<UNK>']) for char in word[:max_word_len]] + 
                        [char_to_index['<PAD>']] * (max_word_len - len(word)) for word in sentence]
        tag_indices = [tag_to_index[tag] for tag in tag_seq]
        
        padded_words = [word_to_index['<PAD>']] * half_window + word_indices + [word_to_index['<PAD>']] * half_window
        padded_chars = [[char_to_index['<PAD>']] * max_word_len] * half_window + char_indices + [[char_to_index['<PAD>']] * max_word_len] * half_window
        
        for i in range(len(sentence)):
            X_words.append(padded_words[i:i + window_size])
            X_chars.append(padded_chars[i:i + window_size])
            y.append(tag_indices[i])
    
    return (torch.tensor(X_words, dtype=torch.long),
            torch.tensor(X_chars, dtype=torch.long),
            torch.tensor(y, dtype=torch.long))

def prepare_test_data_with_chars(sentences, word_to_index, char_to_index, window_size, max_word_len):
    X_words, X_chars = [], []
    half_window = window_size // 2
    for sentence in sentences:
        word_indices = [word_to_index.get(word.lower(), word_to_index['<UNK>']) for word in sentence]
        char_indices = [[char_to_index.get(char, char_to_index['<UNK>']) for char in word[:max_word_len]] + 
                        [char_to_index['<PAD>']] * (max_word_len - len(word)) for word in sentence]
        
        padded_words = [word_to_index['<PAD>']] * half_window + word_indices + [word_to_index['<PAD>']] * half_window
        padded_chars = [[char_to_index['<PAD>']] * max_word_len] * half_window + char_indices + [[char_to_index['<PAD>']] * max_word_len] * half_window
        
        for i in range(len(sentence)):
            X_words.append(padded_words[i:i + window_size])
            X_chars.append(padded_chars[i:i + window_size])
    
    return (torch.tensor(X_words, dtype=torch.long),
            torch.tensor(X_chars, dtype=torch.long))

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total * 100
    return accuracy
import time
def train_model_collect_metrics(train_loader, dev_data, model, criterion, optimizer, scaler, num_epochs, accumulation_steps=4):
    train_losses, train_accuracies = [], []
    dev_losses, dev_accuracies = [], []
    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_accuracy = 0, 0
        epoch_start_time = time.time()

        for i, (batch_words, batch_chars, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch_words, batch_chars, batch_y = batch_words.to(device), batch_chars.to(device), batch_y.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(batch_words, batch_chars)
                loss = criterion(outputs, batch_y)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            epoch_accuracy += calculate_accuracy(outputs, batch_y)

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_accuracy = epoch_accuracy / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        model.eval()
        with torch.no_grad():
            dev_outputs = model(*dev_data[:-1])
            dev_loss = criterion(dev_outputs, dev_data[-1])
            dev_accuracy = calculate_accuracy(dev_outputs, dev_data[-1])
            dev_losses.append(dev_loss.item())
            dev_accuracies.append(dev_accuracy)

        epoch_duration = time.time() - epoch_start_time
        estimated_total_time = epoch_duration * num_epochs
        estimated_remaining_time = estimated_total_time - (epoch_duration * (epoch + 1))

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.2f}%, '
              f'Dev Loss: {dev_loss.item():.4f}, Dev Accuracy: {dev_accuracy:.2f}%, '
              f'Epoch Time: {epoch_duration:.2f}s, '
              f'Estimated Total Time: {estimated_total_time:.2f}s, '
              f'Estimated Remaining Time: {estimated_remaining_time:.2f}s')

    total_training_time = time.time() - total_start_time
    print(f'Total Training Time: {total_training_time:.2f}s')

    return train_losses, train_accuracies, dev_losses, dev_accuracies
# Function to save predictions
def save_predictions(model, X_test, index_to_tag, output_file, test_sentences):
    model.eval()
    with torch.no_grad():
        outputs = model(*X_test)
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

# Plotting function
def plot_metrics(train_losses, train_accuracies, dev_losses, dev_accuracies, title):
    epochs = range(1, len(train_losses) + 1)

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

# Main execution
# Main execution
def analyze_filters(model, char_to_index, top_n=10):
    # Get the weights of the convolutional layer
    conv_weights = model.char_cnn.conv.weight.data.cpu().numpy()
    
    # Get the character embedding
    char_embedding = model.char_cnn.char_embedding.weight.data.cpu().numpy()
    
    num_filters, _, kernel_size = conv_weights.shape
    
    # Create a reverse mapping of char_to_index
    index_to_char = {v: k for k, v in char_to_index.items()}
    
    filter_analysis = []
    
    for i in range(num_filters):
        filter_weights = conv_weights[i]
        
        # Compute the dot product between the filter and all possible character trigrams
        trigram_scores = {}
        for c1 in range(len(char_to_index)):
            for c2 in range(len(char_to_index)):
                for c3 in range(len(char_to_index)):
                    trigram = np.concatenate([char_embedding[c1], char_embedding[c2], char_embedding[c3]])
                    score = np.dot(filter_weights.flatten(), trigram)
                    char_trigram = index_to_char[c1] + index_to_char[c2] + index_to_char[c3]
                    trigram_scores[char_trigram] = score
        
        # Get the top N trigrams for this filter
        top_trigrams = sorted(trigram_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        filter_analysis.append(top_trigrams)
    
    return filter_analysis

def print_filter_analysis(filter_analysis, task_name):
    print(f"\nFilter Analysis for {task_name}:")
    for i, filter_trigrams in enumerate(filter_analysis):
        print(f"Filter {i}:")
        for trigram, score in filter_trigrams:
            print(f"  {trigram}: {score:.4f}")
        print()
if __name__ == "__main__":
    # Load data
    pos_train_sentences, pos_train_tags = load_data('pos/train')
    pos_dev_sentences, pos_dev_tags = load_data('pos/dev')
    pos_test_sentences = load_test_data('pos/test')
    ner_train_sentences, ner_train_tags = load_data('ner/train')
    ner_dev_sentences, ner_dev_tags = load_data('ner/dev')
    ner_test_sentences = load_test_data('ner/test')

    # Build vocabularies
    word_to_index, index_to_word = build_vocab(pos_train_sentences + ner_train_sentences, vocab)
    char_to_index = build_char_vocab(pos_train_sentences + ner_train_sentences)
    pos_tag_to_index, pos_index_to_tag = build_tagset(pos_train_tags)
    ner_tag_to_index, ner_index_to_tag = build_tagset(ner_train_tags)

    print("Vocabulary size:", len(word_to_index))
    print("Character vocabulary size:", len(char_to_index))
    print("POS Tag set size:", len(pos_tag_to_index))
    print("NER Tag set size:", len(ner_tag_to_index))

    # Parameters
    window_size = 5
    max_word_len = 10
    embedding_dim = vectors.shape[1]
    char_embedding_dim = 30
    num_filters = 30
    kernel_size = 3
    hidden_dim = 50
    batch_size = 256
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable CuDNN benchmarking
    torch.backends.cudnn.benchmark = True

    # Prepare pre-trained embeddings
    pre_trained_embeddings = np.zeros((len(word_to_index), embedding_dim))
    for word, idx in word_to_index.items():
        if word in word_to_vec:
            pre_trained_embeddings[idx] = word_to_vec[word]
        else:
            pre_trained_embeddings[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

    # Prepare data
    X_train_pos = prepare_data_with_chars(pos_train_sentences, pos_train_tags, word_to_index, char_to_index, pos_tag_to_index, window_size, max_word_len)
    X_dev_pos = prepare_data_with_chars(pos_dev_sentences, pos_dev_tags, word_to_index, char_to_index, pos_tag_to_index, window_size, max_word_len)
    X_test_pos = prepare_test_data_with_chars(pos_test_sentences, word_to_index, char_to_index, window_size, max_word_len)

    X_train_ner = prepare_data_with_chars(ner_train_sentences, ner_train_tags, word_to_index, char_to_index, ner_tag_to_index, window_size, max_word_len)
    X_dev_ner = prepare_data_with_chars(ner_dev_sentences, ner_dev_tags, word_to_index, char_to_index, ner_tag_to_index, window_size, max_word_len)
    X_test_ner = prepare_test_data_with_chars(ner_test_sentences, word_to_index, char_to_index, window_size, max_word_len)

    # Create DataLoaders
    train_dataset_pos = TaggingDataset(X_train_pos[0], X_train_pos[1], X_train_pos[2])
    train_loader_pos = DataLoader(train_dataset_pos, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    train_dataset_ner = TaggingDataset(X_train_ner[0], X_train_ner[1], X_train_ner[2])
    train_loader_ner = DataLoader(train_dataset_ner, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Move dev and test data to device
    X_dev_pos = tuple(t.to(device) for t in X_dev_pos)
    X_test_pos = tuple(t.to(device) for t in X_test_pos)
    X_dev_ner = tuple(t.to(device) for t in X_dev_ner)
    X_test_ner = tuple(t.to(device) for t in X_test_ner)

    # Train POS model
    pos_model = WindowBasedTaggerWithCNN(
        len(word_to_index), len(char_to_index), char_embedding_dim, num_filters, kernel_size,
        embedding_dim, window_size, hidden_dim=hidden_dim, output_dim=len(pos_tag_to_index),
        pre_trained_embeddings=pre_trained_embeddings
    ).to(device)

    pos_criterion = nn.CrossEntropyLoss()
    pos_optimizer = optim.Adam(pos_model.parameters(), lr=0.001)
    pos_scaler = torch.cuda.amp.GradScaler()

    print("Training POS model...")
    pos_train_losses, pos_train_accuracies, pos_dev_losses, pos_dev_accuracies = train_model_collect_metrics(
        train_loader_pos, X_dev_pos, pos_model, pos_criterion, pos_optimizer, pos_scaler, num_epochs
    )

    # Plot metrics for POS model
    plot_metrics(pos_train_losses, pos_train_accuracies, pos_dev_losses, pos_dev_accuracies, 'POS Model')

    # Analyze filters for POS model
    pos_filter_analysis = analyze_filters(pos_model, char_to_index)
    print_filter_analysis(pos_filter_analysis, "POS Tagging")

    # Save POS test predictions
    save_predictions(pos_model, X_test_pos, pos_index_to_tag, 'test_pos_cnn.txt', pos_test_sentences)

    # Train NER model
    ner_model = WindowBasedTaggerWithCNN(
        len(word_to_index), len(char_to_index), char_embedding_dim, num_filters, kernel_size,
        embedding_dim, window_size, hidden_dim=hidden_dim, output_dim=len(ner_tag_to_index),
        pre_trained_embeddings=pre_trained_embeddings
    ).to(device)

    ner_criterion = nn.CrossEntropyLoss()
    ner_optimizer = optim.Adam(ner_model.parameters(), lr=0.001)
    ner_scaler = torch.cuda.amp.GradScaler()

    print("Training NER model...")
    ner_train_losses, ner_train_accuracies, ner_dev_losses, ner_dev_accuracies = train_model_collect_metrics(
        train_loader_ner, X_dev_ner, ner_model, ner_criterion, ner_optimizer, ner_scaler, num_epochs
    )

    # Plot metrics for NER model
    plot_metrics(ner_train_losses, ner_train_accuracies, ner_dev_losses, ner_dev_accuracies, 'NER Model')

    # Analyze filters for NER model
    ner_filter_analysis = analyze_filters(ner_model, char_to_index)
    print_filter_analysis(ner_filter_analysis, "NER")

    # Save NER test predictions
    save_predictions(ner_model, X_test_ner, ner_index_to_tag, 'test_ner_cnn.txt', ner_test_sentences)

    # Print final results
    print("\nFinal Results:")
    print("POS Tagging:")
    print(f"model parameters:{sum(p.numel() for p in pos_model.parameters())}")
    print(f"Train Accuracy: {pos_train_accuracies[-1]:.2f}%")
    print(f"Dev Accuracy: {pos_dev_accuracies[-1]:.2f}%")
    print("\nNER:")
    print(f"Train Accuracy: {ner_train_accuracies[-1]:.2f}%")
    print(f"Dev Accuracy: {ner_dev_accuracies[-1]:.2f}%")