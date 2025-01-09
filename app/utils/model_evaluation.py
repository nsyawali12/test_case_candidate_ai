import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

def evaluate_lightgbm():
    # Load the test dataset
    test_data_path = "app/utils/results/dummy_test_features_with_GT.csv"  # Update this path if needed
    test_df = pd.read_csv(test_data_path)

    # Check for required columns
    required_columns = ["cosine_similarity", "jaccard_similarity", "levenshtein_similarity", "plagiarism_score"]
    if not all(col in test_df.columns for col in required_columns):
        raise ValueError(f"Test dataset must include the following columns: {required_columns}")

    # Prepare features and ground truth
    X_test = test_df[["cosine_similarity", "jaccard_similarity", "levenshtein_similarity"]]
    true_scores = test_df["plagiarism_score"]

    # Load the trained LightGBM model
    with open("app/models/lightgbm_model_100r.pkl", "rb") as file:
        lgbm_model = pickle.load(file)

    # Predict plagiarism scores
    predictions = lgbm_model.predict(X_test)

    # Metrics
    mse = mean_squared_error(true_scores, predictions)
    mae = mean_absolute_error(true_scores, predictions)
    r2 = r2_score(true_scores, predictions)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Visualization: True vs. Predicted Scores
    plt.figure(figsize=(8, 6))
    plt.scatter(true_scores, predictions, alpha=0.7, label="Predicted")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Ideal Line")
    plt.title("LightGBM: True vs. Predicted Plagiarism Scores")
    plt.xlabel("True Plagiarism Score")
    plt.ylabel("Predicted Plagiarism Score")
    plt.legend()
    plt.grid()
    plt.show()

    # Residual Plot
    residuals = true_scores - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(true_scores, residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("LightGBM: Residual Plot")
    plt.xlabel("True Plagiarism Score")
    plt.ylabel("Residuals")
    plt.grid()
    plt.show()

    # Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(true_scores, color="blue", label="True", kde=True, stat="density", bins=20)
    sns.histplot(predictions, color="orange", label="Predicted", kde=True, stat="density", bins=20)
    plt.title("LightGBM: Distribution of True and Predicted Plagiarism Scores")
    plt.legend()
    plt.grid()
    plt.show()

    # Save predictions
    test_df["predicted_plagiarism_score"] = predictions
    test_df.to_csv("lightgbm_predictions.csv", index=False)
    print("Predictions and results saved to 'lightgbm_predictions.csv'.")


if __name__ == "__main__":
    evaluate_lightgbm()
