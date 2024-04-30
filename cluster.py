from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained Word2Vec model (this can take some time)
# Note: You might need to download the model file first or adjust the path
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# List of words to cluster
words = ['peace', 'peacefulness', 'anger', 'angry', 'happy', 'happiness', 'sad', 'sadness']

# Extract the word vectors for the given words
word_vectors = np.array([model[word] for word in words if word in model])

# Number of clusters
n_clusters = 4

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(word_vectors)

# Assign each word to a cluster
word_clusters = {}
for i, word in enumerate(words):
    if word in model:
        cluster = kmeans.labels_[i]
        if cluster not in word_clusters:
            word_clusters[cluster] = [word]
        else:
            word_clusters[cluster].append(word)

# Print the clusters
for cluster, words in word_clusters.items():
    print(f"Cluster {cluster}: {', '.join(words)}")

def plot_word_clusters(word_vectors, words, labels):
    """
    Plot the word clusters in a 2D graph using t-SNE for dimensionality reduction.

    Parameters:
    - word_vectors: High-dimensional word vectors.
    - words: A list of words corresponding to the vectors.
    - labels: Cluster labels for each word.
    """
    # Reduce dimensions to 2D using t-SNE
    perplexity_value = min(30, len(word_vectors) - 1)  # Ensure perplexity is not more than n_samples - 1
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
    vectors_2d = tsne.fit_transform(word_vectors)

    # Plotting
    plt.figure(figsize=(10, 8))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown'
    for i, word in enumerate(words):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], c=colors[labels[i] % len(colors)])
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title('Word Clusters')
    plt.show()

# Assuming model, words, word_vectors, and kmeans from the previous example are already defined

# Assign each word to a cluster and get labels
labels = kmeans.labels_

# Plot the clustered words
plot_word_clusters(word_vectors, words, labels)