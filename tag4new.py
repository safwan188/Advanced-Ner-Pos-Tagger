import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Data loading functions
def load_data(file_path):
    sentences, tags = [], []
    with open(file_path, 'r') as file:
        sentence, tag_seq = [], []
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence, tag_seq = [], []
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
                sentence.append(line)
        if sentence:
            sentences.append(sentence)
    print(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences

# Vocabulary building functions
def build_vocab(sentences, pre_trained_vocab):
    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    for word in pre_trained_vocab:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
    return word_to_index

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
    for tag_seq in tags:
        for tag in tag_seq:
            if tag not in tag_to_index:
                tag_to_index[tag] = len(tag_to_index)
    return tag_to_index

# Dataset class
class TaggerDataset(Dataset):
    def __init__(self, sentences, tags, word_to_index, char_to_index, tag_to_index, window_size, max_word_len):
        self.sentences = sentences
        self.tags = tags
        self.word_to_index = word_to_index
        self.char_to_index = char_to_index
        self.tag_to_index = tag_to_index
        self.window_size = window_size
        self.max_word_len = max_word_len

    def __len__(self):
        return sum(len(s) for s in self.sentences)

    def __getitem__(self, idx):
        for sentence, tag_seq in zip(self.sentences, self.tags):
            if idx < len(sentence):
                break
            idx -= len(sentence)
        
        half_window = self.window_size // 2
        padded_sentence = ['<PAD>'] * half_window + sentence + ['<PAD>'] * half_window
        
        words = [self.word_to_index.get(w.lower(), self.word_to_index['<UNK>']) for w in padded_sentence[idx:idx+self.window_size]]
        chars = [[self.char_to_index.get(c, self.char_to_index['<UNK>']) for c in w[:self.max_word_len]] +
                 [self.char_to_index['<PAD>']] * (self.max_word_len - len(w)) for w in padded_sentence[idx:idx+self.window_size]]
        tag = self.tag_to_index[tag_seq[idx]]
        chars = [[self.char_to_index.get(c, self.char_to_index['<UNK>']) for c in w[:self.max_word_len]] +
             [self.char_to_index['<PAD>']] * (self.max_word_len - len(w)) for w in padded_sentence[idx:idx+self.window_size]]
    
        return torch.tensor(words), torch.tensor(chars), torch.tensor(tag)

# Model definition
class WindowBasedTaggerWithCharCNN(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, embedding_dim, char_embedding_dim, 
                 window_size, hidden_dim, output_dim, num_filters, kernel_size, pre_trained_embeddings=None):
        super(WindowBasedTaggerWithCharCNN, self).__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        if pre_trained_embeddings is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pre_trained_embeddings))
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_cnn = nn.Conv1d(char_embedding_dim, num_filters, kernel_size, padding=kernel_size-1)
        
        self.fc1 = nn.Linear(window_size * (embedding_dim + num_filters), hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, words, chars):
        word_embeds = self.word_embedding(words)
        
        batch_size, seq_len, char_seq_len = chars.size()
        chars = chars.view(batch_size * seq_len, char_seq_len)
        char_embeds = self.char_embedding(chars)
        char_embeds = char_embeds.transpose(1, 2)  # (batch_size * seq_len, char_embedding_dim, char_seq_len)
        char_feature = self.char_cnn(char_embeds)
        char_feature = torch.max(char_feature, dim=2)[0]  # (batch_size * seq_len, num_filters)
        char_feature = char_feature.view(batch_size, seq_len, -1)
        
        combined_embeds = torch.cat((word_embeds, char_feature), dim=-1)
        
        embeds = combined_embeds.view((words.size(0), -1))
        hidden = self.tanh(self.fc1(embeds))
        output = self.fc2(hidden)
        return output

    def get_char_cnn_filters(self):
        return self.char_cnn.weight.data.cpu().numpy()

# Training function
def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs, patience, device):
    best_dev_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses = []
    dev_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for words, chars, tags in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            words, chars, tags = words.to(device), chars.to(device), tags.to(device)
            
            optimizer.zero_grad()
            outputs = model(words, chars)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for words, chars, tags in dev_loader:
                words, chars, tags = words.to(device), chars.to(device), tags.to(device)
                outputs = model(words, chars)
                loss = criterion(outputs, tags)
                total_dev_loss += loss.item()
        
        avg_dev_loss = total_dev_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")
        
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                model.load_state_dict(torch.load('best_model.pth'))
                break
    
    return train_losses, dev_losses

# Prediction function
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for words, chars, _ in test_loader:
            words, chars = words.to(device), chars.to(device)
            outputs = model(words, chars)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# Main execution
if __name__ == "__main__":
    # Load data
    pos_train_sentences, pos_train_tags = load_data('pos/train')
    pos_dev_sentences, pos_dev_tags = load_data('pos/dev')
    pos_test_sentences = load_test_data('pos/test')
    ner_train_sentences, ner_train_tags = load_data('ner/train')
    ner_dev_sentences, ner_dev_tags = load_data('ner/dev')
    ner_test_sentences = load_test_data('ner/test')

    # Load pre-trained embeddings
    vectors = np.loadtxt("wordVectors.txt")
    with open("vocab.txt", "r") as f:
        vocab = [line.strip() for line in f]

    # Build vocabularies
    word_to_index = build_vocab(pos_train_sentences + ner_train_sentences, vocab)
    char_to_index = build_char_vocab(pos_train_sentences + ner_train_sentences)
    pos_tag_to_index = build_tagset(pos_train_tags)
    ner_tag_to_index = build_tagset(ner_train_tags)

    # Set parameters
    window_size = 5
    embedding_dim = vectors.shape[1]
    char_embedding_dim = 30
    num_filters = 30
    kernel_size = 3
    max_word_len = 20
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 50
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare pre-trained embeddings
    pre_trained_embeddings = np.zeros((len(word_to_index), embedding_dim))
    for word, idx in word_to_index.items():
        if word in vocab:
            pre_trained_embeddings[idx] = vectors[vocab.index(word)]
        else:
            pre_trained_embeddings[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

    # Prepare datasets and dataloaders
    pos_train_dataset = TaggerDataset(pos_train_sentences, pos_train_tags, word_to_index, char_to_index, pos_tag_to_index, window_size, max_word_len)
    pos_dev_dataset = TaggerDataset(pos_dev_sentences, pos_dev_tags, word_to_index, char_to_index, pos_tag_to_index, window_size, max_word_len)
    pos_test_dataset = TaggerDataset(pos_test_sentences, [['O']*len(s) for s in pos_test_sentences], word_to_index, char_to_index, pos_tag_to_index, window_size, max_word_len)

    ner_train_dataset = TaggerDataset(ner_train_sentences, ner_train_tags, word_to_index, char_to_index, ner_tag_to_index, window_size, max_word_len)
    ner_dev_dataset = TaggerDataset(ner_dev_sentences, ner_dev_tags, word_to_index, char_to_index, ner_tag_to_index, window_size, max_word_len)
    ner_test_dataset = TaggerDataset(ner_test_sentences, [['O']*len(s) for s in ner_test_sentences], word_to_index, char_to_index, ner_tag_to_index, window_size, max_word_len)

    pos_train_loader = DataLoader(pos_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    pos_dev_loader = DataLoader(pos_dev_dataset, batch_size=batch_size, num_workers=4)
    pos_test_loader = DataLoader(pos_test_dataset, batch_size=batch_size, num_workers=4)

    ner_train_loader = DataLoader(ner_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    ner_dev_loader = DataLoader(ner_dev_dataset, batch_size=batch_size, num_workers=4)
    ner_test_loader = DataLoader(ner_test_dataset, batch_size=batch_size, num_workers=4)

    # Train POS model
    pos_model = WindowBasedTaggerWithCharCNN(len(word_to_index), len(char_to_index), embedding_dim, 
                                             char_embedding_dim, window_size, 50, len(pos_tag_to_index), 
                                             num_filters, kernel_size, pre_trained_embeddings).to(device)
    print(pos_model)
    pos_criterion = nn.CrossEntropyLoss()
    pos_optimizer = optim.Adam(pos_model.parameters(), lr=learning_rate)

    print("Training POS model...")
    pos_train_losses, pos_dev_losses = train_model(pos_model, pos_train_loader, pos_dev_loader, pos_criterion, pos_optimizer, num_epochs, patience, device)

    # Train NER model
    ner_model = WindowBasedTaggerWithCharCNN(len(word_to_index), len(char_to_index), embedding_dim, 
                                             char_embedding_dim, window_size, 10, len(ner_tag_to_index), 
                                             num_filters, kernel_size, pre_trained_embeddings).to(device)
    ner_criterion = nn.CrossEntropyLoss()
    ner_optimizer = optim.Adam(ner_model.parameters(), lr=learning_rate)

    print("Training NER model...")
    ner_train_losses, ner_dev_losses = train_model(ner_model, ner_train_loader, ner_dev_loader, ner_criterion, ner_optimizer, num_epochs, patience, device)

    # Make predictions
    pos_predictions = predict(pos_model, pos_test_loader, device)
    ner_predictions = predict(ner_model, ner_test_loader, device)

    # Save predictions
    def save_predictions(predictions, sentences, index_to_tag, output_file):
        with open(output_file, 'w') as f:
            pred_idx = 0
            for sentence in sentences:
                for word in sentence:
                    pred_tag = index_to_tag[predictions[pred_idx]]
                    f.write(f"{word} {pred_tag}\n")
                    pred_idx += 1
                f.write("\n")
        print(f"Predictions saved to {output_file}")

    pos_index_to_tag = {v: k for k, v in pos_tag_to_index.items()}
    ner_index_to_tag = {v: k for k, v in ner_tag_to_index.items()}

    save_predictions(pos_predictions, pos_test_sentences, pos_index_to_tag, 'test4.pos')
    save_predictions(ner_predictions, ner_test_sentences, ner_index_to_tag, 'test4.ner')

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(pos_train_losses, label='Train Loss')
    plt.plot(pos_dev_losses, label='Dev Loss')
    plt.title('POS Tagging Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ner_train_losses, label='Train Loss')
    plt.plot(ner_dev_losses, label='Dev Loss')
    plt.title('NER Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

    # Analyze CNN filters
    def analyze_cnn_filters(model, dataset, num_top_words=10):
        model.eval()
        filters = model.get_char_cnn_filters()
        num_filters, _, kernel_size = filters.shape

        # Get all unique words from the dataset
        all_words = set()
        for sentence in dataset.sentences:
            all_words.update(sentence)

        # Compute activations for each word
        word_activations = {}
        for word in tqdm(all_words, desc="Computing filter activations"):
            chars = torch.tensor([[dataset.char_to_index.get(c, dataset.char_to_index['<UNK>']) for c in word]])
            with torch.no_grad():
                char_embeds = model.char_embedding(chars.to(device))
                char_embeds = char_embeds.transpose(1, 2)
                activations = model.char_cnn(char_embeds)
                max_activations = torch.max(activations, dim=2)[0].squeeze()
            word_activations[word] = max_activations.cpu().numpy()

        # Find top words for each filter
        top_words_per_filter = []
        for i in range(num_filters):
            sorted_words = sorted(word_activations.keys(), key=lambda w: word_activations[w][i], reverse=True)
            top_words_per_filter.append(sorted_words[:num_top_words])

        return top_words_per_filter

    print("Analyzing POS CNN filters...")
    pos_top_words = analyze_cnn_filters(pos_model, pos_train_dataset)

    print("Analyzing NER CNN filters...")
    ner_top_words = analyze_cnn_filters(ner_model, ner_train_dataset)

    # Save filter analysis results
    def save_filter_analysis(top_words, output_file):
        with open(output_file, 'w') as f:
            for i, words in enumerate(top_words):
                f.write(f"Filter {i}:\n")
                for word in words:
                    f.write(f"  {word}\n")
                f.write("\n")
        print(f"Filter analysis saved to {output_file}")

    save_filter_analysis(pos_top_words, 'pos_filter_analysis.txt')
    save_filter_analysis(ner_top_words, 'ner_filter_analysis.txt')

    # Compare POS and NER filters
    def compare_filters(pos_top_words, ner_top_words):
        num_filters = len(pos_top_words)
        similarity_matrix = np.zeros((num_filters, num_filters))

        for i in range(num_filters):
            for j in range(num_filters):
                pos_set = set(pos_top_words[i])
                ner_set = set(ner_top_words[j])
                jaccard_similarity = len(pos_set.intersection(ner_set)) / len(pos_set.union(ner_set))
                similarity_matrix[i, j] = jaccard_similarity

        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('POS vs NER Filter Similarity')
        plt.xlabel('NER Filters')
        plt.ylabel('POS Filters')
        plt.savefig('filter_similarity.png')
        plt.close()

    print("Comparing POS and NER filters...")
    compare_filters(pos_top_words, ner_top_words)

    print("Analysis complete. Check the output files and images for results.")

    # Free up memory
    del pos_model, ner_model, pos_train_loader, pos_dev_loader, pos_test_loader, ner_train_loader, ner_dev_loader, ner_test_loader
    gc.collect()
    torch.cuda.empty_cache()
