import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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

def build_affix_vocab(sentences):
    prefix_to_index = {'<UNK>': 0}
    suffix_to_index = {'<UNK>': 0}
    for sentence in sentences:
        for word in sentence:
            if len(word) >= 3:
                prefix = word[:3].lower()
                suffix = word[-3:].lower()
                if prefix not in prefix_to_index:
                    prefix_to_index[prefix] = len(prefix_to_index)
                if suffix not in suffix_to_index:
                    suffix_to_index[suffix] = len(suffix_to_index)
    return prefix_to_index, suffix_to_index

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
class WindowBasedTaggerWithAffixes(nn.Module):
    def __init__(self, vocab_size, prefix_size, suffix_size, embedding_dim, window_size, hidden_dim, output_dim, pre_trained_embeddings=None):
        super(WindowBasedTaggerWithAffixes, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        if pre_trained_embeddings is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pre_trained_embeddings))
        self.prefix_embedding = nn.Embedding(prefix_size, embedding_dim)
        self.suffix_embedding = nn.Embedding(suffix_size, embedding_dim)
        self.fc1 = nn.Linear(window_size * embedding_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, words, prefixes, suffixes):
        word_embeds = self.word_embedding(words)
        prefix_embeds = self.prefix_embedding(prefixes)
        suffix_embeds = self.suffix_embedding(suffixes)
        combined_embeds = word_embeds + prefix_embeds + suffix_embeds
        embeds = combined_embeds.view((words.size(0), -1))
        hidden = self.tanh(self.fc1(embeds))
        output = self.fc2(hidden)
        return output

# Data preparation function
def prepare_data_with_affixes(sentences, tags, word_to_index, prefix_to_index, suffix_to_index, tag_to_index, window_size):
    X_words, X_prefixes, X_suffixes, y = [], [], [], []
    half_window = window_size // 2
    for sentence, tag_seq in zip(sentences, tags):
        word_indices = [word_to_index.get(word.lower(), word_to_index['<UNK>']) for word in sentence]
        prefix_indices = [prefix_to_index.get(word[:3].lower(), prefix_to_index['<UNK>']) if len(word) >= 3 else prefix_to_index['<UNK>'] for word in sentence]
        suffix_indices = [suffix_to_index.get(word[-3:].lower(), suffix_to_index['<UNK>']) if len(word) >= 3 else suffix_to_index['<UNK>'] for word in sentence]
        tag_indices = [tag_to_index[tag] for tag in tag_seq]
        
        padded_words = [word_to_index['<PAD>']] * half_window + word_indices + [word_to_index['<PAD>']] * half_window
        padded_prefixes = [prefix_to_index['<UNK>']] * half_window + prefix_indices + [prefix_to_index['<UNK>']] * half_window
        padded_suffixes = [suffix_to_index['<UNK>']] * half_window + suffix_indices + [suffix_to_index['<UNK>']] * half_window
        
        for i in range(len(sentence)):
            X_words.append(padded_words[i:i + window_size])
            X_prefixes.append(padded_prefixes[i:i + window_size])
            X_suffixes.append(padded_suffixes[i:i + window_size])
            y.append(tag_indices[i])
    
    return (torch.tensor(X_words, dtype=torch.long),
            torch.tensor(X_prefixes, dtype=torch.long),
            torch.tensor(X_suffixes, dtype=torch.long),
            torch.tensor(y, dtype=torch.long))

def prepare_test_data_with_affixes(sentences, word_to_index, prefix_to_index, suffix_to_index, window_size):
    X_words, X_prefixes, X_suffixes = [], [], []
    half_window = window_size // 2
    for sentence in sentences:
        word_indices = [word_to_index.get(word.lower(), word_to_index['<UNK>']) for word in sentence]
        prefix_indices = [prefix_to_index.get(word[:3].lower(), prefix_to_index['<UNK>']) if len(word) >= 3 else prefix_to_index['<UNK>'] for word in sentence]
        suffix_indices = [suffix_to_index.get(word[-3:].lower(), suffix_to_index['<UNK>']) if len(word) >= 3 else suffix_to_index['<UNK>'] for word in sentence]
        
        padded_words = [word_to_index['<PAD>']] * half_window + word_indices + [word_to_index['<PAD>']] * half_window
        padded_prefixes = [prefix_to_index['<UNK>']] * half_window + prefix_indices + [prefix_to_index['<UNK>']] * half_window
        padded_suffixes = [suffix_to_index['<UNK>']] * half_window + suffix_indices + [suffix_to_index['<UNK>']] * half_window
        
        for i in range(len(sentence)):
            X_words.append(padded_words[i:i + window_size])
            X_prefixes.append(padded_prefixes[i:i + window_size])
            X_suffixes.append(padded_suffixes[i:i + window_size])
    
    return (torch.tensor(X_words, dtype=torch.long),
            torch.tensor(X_prefixes, dtype=torch.long),
            torch.tensor(X_suffixes, dtype=torch.long))

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total * 100
    return accuracy

# Function to train model and collect metrics
def train_model_collect_metrics(X_train, y_train, X_dev, y_dev, vocab_size, prefix_size, suffix_size, output_dim, learning_rate, hidden_dim, num_epochs, pre_trained_embeddings=None):
    model = WindowBasedTaggerWithAffixes(vocab_size, prefix_size, suffix_size, embedding_dim, window_size, hidden_dim, output_dim, pre_trained_embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    dev_losses = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(*X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        train_accuracy = calculate_accuracy(train_outputs, y_train)
        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy)
        
        model.eval()
        with torch.no_grad():
            dev_outputs = model(*X_dev)
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

# Function to analyze OOV performance
def analyze_oov_performance(model, X_dev, y_dev, word_to_index, index_to_word):
    model.eval()
    with torch.no_grad():
        outputs = model(*X_dev)
        _, predicted = torch.max(outputs, 1)
    
    oov_correct = 0
    oov_total = 0
    for i, word_idx in enumerate(X_dev[0][:, window_size // 2]):  # Center word of each window
        if index_to_word[word_idx.item()] == '<UNK>':
            oov_total += 1
            if predicted[i] == y_dev[i]:
                oov_correct += 1
    
    oov_accuracy = (oov_correct / oov_total * 100) if oov_total > 0 else 0
    return oov_accuracy

# Main execution
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
    prefix_to_index, suffix_to_index = build_affix_vocab(pos_train_sentences + ner_train_sentences)
    pos_tag_to_index, pos_index_to_tag = build_tagset(pos_train_tags)
    ner_tag_to_index, ner_index_to_tag = build_tagset(ner_train_tags)

    print("Vocabulary size:", len(word_to_index))
    print("Prefix vocabulary size:", len(prefix_to_index))
    print("Suffix vocabulary size:", len(suffix_to_index))
    print("POS Tag set size:", len(pos_tag_to_index))
    print("NER Tag set size:", len(ner_tag_to_index))

    # Parameters
    window_size = 5
    embedding_dim = vectors.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare pre-trained embeddings
    pre_trained_embeddings = np.zeros((len(word_to_index), embedding_dim))
    for word, idx in word_to_index.items():
        if word in word_to_vec:
            pre_trained_embeddings[idx] = word_to_vec[word]
        else:
            # Prepare pre-trained embeddings
            pre_trained_embeddings = np.zeros((len(word_to_index), embedding_dim))
    for word, idx in word_to_index.items():
        if word in word_to_vec:
            pre_trained_embeddings[idx] = word_to_vec[word]
        else:
            pre_trained_embeddings[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

    # Prepare data
    X_train_pos = prepare_data_with_affixes(pos_train_sentences, pos_train_tags, word_to_index, prefix_to_index, suffix_to_index, pos_tag_to_index, window_size)
    X_dev_pos = prepare_data_with_affixes(pos_dev_sentences, pos_dev_tags, word_to_index, prefix_to_index, suffix_to_index, pos_tag_to_index, window_size)
    X_test_pos = prepare_test_data_with_affixes(pos_test_sentences, word_to_index, prefix_to_index, suffix_to_index, window_size)

    X_train_ner = prepare_data_with_affixes(ner_train_sentences, ner_train_tags, word_to_index, prefix_to_index, suffix_to_index, ner_tag_to_index, window_size)
    X_dev_ner = prepare_data_with_affixes(ner_dev_sentences, ner_dev_tags, word_to_index, prefix_to_index, suffix_to_index, ner_tag_to_index, window_size)
    X_test_ner = prepare_test_data_with_affixes(ner_test_sentences, word_to_index, prefix_to_index, suffix_to_index, window_size)

    # Move data to device
    X_train_pos = tuple(t.to(device) for t in X_train_pos)
    X_dev_pos = tuple(t.to(device) for t in X_dev_pos)
    X_test_pos = tuple(t.to(device) for t in X_test_pos)
    X_train_ner = tuple(t.to(device) for t in X_train_ner)
    X_dev_ner = tuple(t.to(device) for t in X_dev_ner)
    X_test_ner = tuple(t.to(device) for t in X_test_ner)

    # Train POS model with pre-trained embeddings
    best_pos_params = {'learning_rate': 0.005, 'hidden_dim': 50}
    num_epochs = 200
    print("Training POS model with pre-trained embeddings...")
    pos_model_pre, pos_train_losses_pre, pos_train_accuracies_pre, pos_dev_losses_pre, pos_dev_accuracies_pre = train_model_collect_metrics(
        X_train_pos[:-1], X_train_pos[-1], X_dev_pos[:-1], X_dev_pos[-1],
        len(word_to_index), len(prefix_to_index), len(suffix_to_index), len(pos_tag_to_index),
        best_pos_params['learning_rate'], best_pos_params['hidden_dim'], num_epochs, pre_trained_embeddings
    )

    # Train POS model without pre-trained embeddings
    print("Training POS model without pre-trained embeddings...")
    pos_model_no_pre, pos_train_losses_no_pre, pos_train_accuracies_no_pre, pos_dev_losses_no_pre, pos_dev_accuracies_no_pre = train_model_collect_metrics(
        X_train_pos[:-1], X_train_pos[-1], X_dev_pos[:-1], X_dev_pos[-1],
        len(word_to_index), len(prefix_to_index), len(suffix_to_index), len(pos_tag_to_index),
        best_pos_params['learning_rate'], best_pos_params['hidden_dim'], num_epochs
    )

    # Plot metrics for POS models
    plot_metrics(pos_train_losses_pre, pos_train_accuracies_pre, pos_dev_losses_pre, pos_dev_accuracies_pre, 'POS Model with Pre-trained Embeddings')
    plot_metrics(pos_train_losses_no_pre, pos_train_accuracies_no_pre, pos_dev_losses_no_pre, pos_dev_accuracies_no_pre, 'POS Model without Pre-trained Embeddings')

    # Save POS test predictions
   # Save POS test predictions
    save_predictions(pos_model_pre, X_test_pos, pos_index_to_tag, 'test3_pre.pos', pos_test_sentences)
    save_predictions(pos_model_no_pre, X_test_pos, pos_index_to_tag, 'test3_no_pre.pos', pos_test_sentences)

    # Train NER model with pre-trained embeddings
    best_ner_params = {'learning_rate': 0.003, 'hidden_dim': 10}
    num_epochs = 150
    print("Training NER model with pre-trained embeddings...")
    ner_model_pre, ner_train_losses_pre, ner_train_accuracies_pre, ner_dev_losses_pre, ner_dev_accuracies_pre = train_model_collect_metrics(
        X_train_ner[:-1], X_train_ner[-1], X_dev_ner[:-1], X_dev_ner[-1],
        len(word_to_index), len(prefix_to_index), len(suffix_to_index), len(ner_tag_to_index),
        best_ner_params['learning_rate'], best_ner_params['hidden_dim'], num_epochs, pre_trained_embeddings
    )

    # Train NER model without pre-trained embeddings
    print("Training NER model without pre-trained embeddings...")
    ner_model_no_pre, ner_train_losses_no_pre, ner_train_accuracies_no_pre, ner_dev_losses_no_pre, ner_dev_accuracies_no_pre = train_model_collect_metrics(
        X_train_ner[:-1], X_train_ner[-1], X_dev_ner[:-1], X_dev_ner[-1],
        len(word_to_index), len(prefix_to_index), len(suffix_to_index), len(ner_tag_to_index),
        best_ner_params['learning_rate'], best_ner_params['hidden_dim'], num_epochs
    )

    # Plot metrics for NER models
    plot_metrics(ner_train_losses_pre, ner_train_accuracies_pre, ner_dev_losses_pre, ner_dev_accuracies_pre, 'NER Model with Pre-trained Embeddings')
    plot_metrics(ner_train_losses_no_pre, ner_train_accuracies_no_pre, ner_dev_losses_no_pre, ner_dev_accuracies_no_pre, 'NER Model without Pre-trained Embeddings')

    # Save NER test predictions
    # Save NER test predictions
    save_predictions(ner_model_pre, X_test_ner, ner_index_to_tag, 'test3_pre.ner', ner_test_sentences)
    save_predictions(ner_model_no_pre, X_test_ner, ner_index_to_tag, 'test3_no_pre.ner', ner_test_sentences)

    # Comparison of models
    print("\nComparison of models:")
    print("POS Tagging:")
    print(f"With pre-trained embeddings: Dev Accuracy: {pos_dev_accuracies_pre[-1]:.2f}%")
    print(f"Without pre-trained embeddings: Dev Accuracy: {pos_dev_accuracies_no_pre[-1]:.2f}%")

    print("\nNER:")
    print(f"With pre-trained embeddings: Dev Accuracy: {ner_dev_accuracies_pre[-1]:.2f}%")
    print(f"Without pre-trained embeddings: Dev Accuracy: {ner_dev_accuracies_no_pre[-1]:.2f}%")

    # Additional analysis for out-of-vocabulary words
    pos_oov_accuracy_pre = analyze_oov_performance(pos_model_pre, X_dev_pos[:-1], X_dev_pos[-1], word_to_index, index_to_word)
    pos_oov_accuracy_no_pre = analyze_oov_performance(pos_model_no_pre, X_dev_pos[:-1], X_dev_pos[-1], word_to_index, index_to_word)
    ner_oov_accuracy_pre = analyze_oov_performance(ner_model_pre, X_dev_ner[:-1], X_dev_ner[-1], word_to_index, index_to_word)
    ner_oov_accuracy_no_pre = analyze_oov_performance(ner_model_no_pre, X_dev_ner[:-1], X_dev_ner[-1], word_to_index, index_to_word)

    print("\nOut-of-vocabulary word performance:")
    print(f"POS Tagging OOV Accuracy (with pre-trained): {pos_oov_accuracy_pre:.2f}%")
    print(f"POS Tagging OOV Accuracy (without pre-trained): {pos_oov_accuracy_no_pre:.2f}%")
    print(f"NER OOV Accuracy (with pre-trained): {ner_oov_accuracy_pre:.2f}%")
    print(f"NER OOV Accuracy (without pre-trained): {ner_oov_accuracy_no_pre:.2f}%")