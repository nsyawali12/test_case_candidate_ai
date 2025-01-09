import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_codebert():
    test_data_path = "app/utils/results/dummy_test_features_with_GT.csv"  # Update this path if needed
    test_df = pd.read_csv(test_data_path)

    required_columns = ["candidate_answer", "ai_generated_answer", "plagiarism_score"]
    if not all(col in test_df.columns for col in required_columns):
        raise ValueError(f"Test dataset must include the following columns: {required_columns}")
    
    # Load the fine-tuned CodeBERT model and tokenizer
    model_path = "app/models/codebert_finetuned_60e"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, from_tf=True)

    # Set model to evaluation mode
    model.eval()

    # Prepare inputs for prediction
    predictions = []
    true_scores = test_df["plagiarism_score"].tolist()

    with torch.no_grad():
        for _, row in test_df.iterrows():
            inputs = tokenizer(
                row["candidate_answer"], row["ai_generated_answer"],
                return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_score = torch.sigmoid(logits).item()  # Convert logits to probabilities
            predictions.append(predicted_score)

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
    plt.title("CodeBERT: True vs. Predicted Plagiarism Scores")
    plt.xlabel("True Plagiarism Score")
    plt.ylabel("Predicted Plagiarism Score")
    plt.legend()
    plt.grid()
    plt.show()

    # Residual Plot
    residuals = [true - pred for true, pred in zip(true_scores, predictions)]
    plt.figure(figsize=(8, 6))
    plt.scatter(true_scores, residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("CodeBERT: Residual Plot")
    plt.xlabel("True Plagiarism Score")
    plt.ylabel("Residuals")
    plt.grid()
    plt.show()

    # Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(true_scores, color="blue", label="True", kde=True, stat="density", bins=20)
    sns.histplot(predictions, color="orange", label="Predicted", kde=True, stat="density", bins=20)
    plt.title("CodeBERT: Distribution of True and Predicted Plagiarism Scores")
    plt.legend()
    plt.grid()
    plt.show()

    # Save predictions
    test_df["predicted_plagiarism_score"] = predictions
    test_df.to_csv("codebert_predictions.csv", index=False)
    print("Predictions and results saved to 'codebert_predictions.csv'.")

if __name__ == "__main__":
    evaluate_codebert()