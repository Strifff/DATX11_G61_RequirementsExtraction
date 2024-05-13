from tqdm import tqdm

import numpy as np
import pickle
import os


class Vectorizer:
    def __init__(self):
        embedding_file = "vector_stuff/glove.42B.300d.txt"
        self.embedding_model = self.load_embeddings(embedding_file)

    def load_embeddings(self, embedding_file):
        pickle_file = embedding_file + ".pkl"
        # Try to load embeddings from pickled file
        if os.path.exists(pickle_file):
            print("Loading embeddings from pickle file...")
            with open(pickle_file, "rb") as f:
                embeddings = pickle.load(f)
            return embeddings
        else:
            print("Loading embeddings from text file...")
            embeddings = {}
            with open(embedding_file, "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    parts = line.split()
                    word = parts[0]
                    embedding = np.array([float(val) for val in parts[1:]])
                    embeddings[word] = embedding
            # Save embeddings to a pickle file
            with open(pickle_file, "wb") as f:
                pickle.dump(embeddings, f)
            return embeddings

    def vectorize_tokens(self, tokenized_sentence):
        # Get embeddings for each token in the tokenized sentence
        token_embeddings = [
            self.embedding_model[token]
            for token in tokenized_sentence
            if token in self.embedding_model
        ]

        # Combine embeddings (for example, by averaging)
        if token_embeddings:
            sentence_embedding = np.mean(token_embeddings, axis=0)
        else:
            # If token_embeddings is empty, return a zero vector of the same size as the embeddings
            sentence_embedding = np.zeros_like(
                next(iter(self.embedding_model.values()))
            )

        return sentence_embedding

    def cosine_distance(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return 1 - dot_product / (norm1 * norm2)

    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)
