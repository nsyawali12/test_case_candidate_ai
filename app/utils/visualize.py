import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_predictions(predict_path):

    df = pd.read_csv(predict_path)

    # Histogram of predicted plagiarism scores
    plt.figure(figsize=(8, 6))
    sns.histplot(df["predicted_plagiarism_score"], kde=True, bins=20, color="skyblue", stat="density")
    plt.title("Distribution of Predicted Plagiarism Scores")
    plt.xlabel("Predicted Plagiarism Score")
    plt.ylabel("Density")
    plt.grid()
    plt.show()

    # Scatter plot of cosine similarity vs predicted plagiarism score
    plt.figure(figsize=(8, 6))
    plt.scatter(df["cosine_similarity"], df["predicted_plagiarism_score"], alpha=0.7, label="Cosine Similarity")
    plt.title("Predicted Plagiarism Score vs. Cosine Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Predicted Plagiarism Score")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(df["jaccard_similarity"], df["predicted_plagiarism_score"], alpha=0.7, label="Jaccard Similarity", color="green")
    plt.title("Predicted Plagiarism Score vs. Jaccard Similarity")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Predicted Plagiarism Score")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(df["levenshtein_similarity"], df["predicted_plagiarism_score"], alpha=0.7, label="Levenshtein Similarity", color="orange")
    plt.title("Predicted Plagiarism Score vs. Levenshtein Similarity")
    plt.xlabel("Levenshtein Similarity")
    plt.ylabel("Predicted Plagiarism Score")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    predictions_file = "app/utils/results/lightgbm_predictions_.csv"
    visualize_predictions(predictions_file)