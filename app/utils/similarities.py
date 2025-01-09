from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
from Levenshtein import distance as levenshtein_distance
import numpy as np

def tokenize_to_sequence(text):
    return [hash(word) % (10**6) for word in text.split()]

def compute_cosine_similarity(seq1, seq2):
    vec1 = np.bincount(seq1, minlength=max(seq1 + seq2) + 1)
    vec2 = np.bincount(seq2, minlength=max(seq1 + seq2) + 1)
    return cosine_similarity([vec1], [vec2])[0][0]

def compute_jaccard_similarity(seq1, seq2):
    set1, set2 = set(seq1), set(seq2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compute_levenshtein_similarity(seq1, seq2):
    seq1_str = " ".join(map(str, seq1))
    seq2_str = " ".join(map(str, seq2))
    return 1 / (1 + levenshtein_distance(seq1_str, seq2_str))

def compute_similarity_metrics(candidate_answer, ai_generated_answer):
    # Tokenize text into sequences
    seq1 = tokenize_to_sequence(candidate_answer)
    seq2 = tokenize_to_sequence(ai_generated_answer)

    # Compute similarity metrics
    cosine = compute_cosine_similarity(seq1, seq2)
    jaccard = compute_jaccard_similarity(seq1, seq2)
    levenshtein = compute_levenshtein_similarity(seq1, seq2)

    return {
        "cosine_similarity": cosine,
        "jaccard_similarity": jaccard,
        "levenshtein_similarity": levenshtein,
    }

if __name__ == "__main__":
    # Example usage
    candidate = "def add_numbers(a, b): return a + b"
    ai_generated = "def add_numbers(x, y): return x + y"
    
    metrics = compute_similarity_metrics(candidate, ai_generated)
    print("Similarity Metrics:", metrics)