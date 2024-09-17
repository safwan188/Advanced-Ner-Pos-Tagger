import numpy as np
from scipy.spatial.distance import cosine

# Load pre-trained embeddings and vocabulary
vectors = np.loadtxt("wordVectors.txt")
with open("vocab.txt", "r") as f:
    vocab = [line.strip() for line in f]

# Create a dictionary mapping words to their vectors
word_to_vec = {word: vector for word, vector in zip(vocab, vectors)}

def cosine_similarity(u, v):
    return 1 - cosine(u, v)

def most_similar(word, k):
    if word not in word_to_vec:
        return []
    
    word_vector = word_to_vec[word]
    similarities = [(w, cosine_similarity(word_vector, vec)) 
                    for w, vec in word_to_vec.items() if w != word]
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:k]

# Words to find similar words for
words = ["dog", "england", "john", "explode", "office"]

for word in words:
    print(f"Most similar words to '{word}':")
    similar_words = most_similar(word, 5)
    for similar_word, similarity in similar_words:
        print(f"  {similar_word}: {similarity:.4f}")
    print()