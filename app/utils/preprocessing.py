import pandas as pd
import ast

def load_and_preprocess(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    
    # Ensure tokenized sequences are properly parsed
    df['candidate_seq'] = df['candidate_seq'].apply(ast.literal_eval)
    df['ai_generated_seq'] = df['ai_generated_seq'].apply(ast.literal_eval)
    
    return df

if __name__ == "__main__":
    fp = "app/utils/similarity_features.csv"
    df = load_and_preprocess(fp)
    print("Preprocessed DataFrame Sample: ")
    print(df.head)