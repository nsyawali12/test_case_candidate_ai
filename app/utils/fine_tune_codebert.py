from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from transformers import TFRobertaForSequenceClassification, RobertaConfig
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import torch

class CodeBERTFineTuner:
    def __init__(self, model_name="microsoft/codebert-base", max_length=512, batch_size=8, learning_rate=5e-5, epochs=10, model_path=None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = TFRobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=1)

        if model_path:
            # Load TensorFlow-specific weights
            config = RobertaConfig.from_pretrained(model_path)
            self.model = TFRobertaForSequenceClassification.from_pretrained(model_path, config=config)
        else:
            # Initialize a new model
            self.model = TFRobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)

        # Set model to evaluation mode
        #self.model.eval()

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        return df
    
    def preprocess_data(self, df):
        # Prepare text and labels
        candidate_text = df['candidate_seq'].tolist()
        ai_texts = df['ai_generated_seq'].tolist()
        labels = df['plagiarism_score'].tolist()

        # Split into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            list(zip(candidate_text, ai_texts)),
            labels,
            test_size=0.2,
            random_state=42
        )

        # Tokenize text pairs
        train_encodings = self.tokenizer.batch_encode_plus(
            [(pair[0], pair[1]) for pair in train_texts],
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="tf"
        )
        val_encodings = self.tokenizer.batch_encode_plus(
            [(pair[0], pair[1]) for pair in val_texts],
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="tf"
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            tf.constant(train_labels, dtype=tf.float32)
        )).batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            tf.constant(val_labels, dtype=tf.float32)
        )).batch(self.batch_size)

        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs
        )
        return history
    
    def save_model(self, output_dir="codebert_finetuned_10e"):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}!")

    def predict(self, candidate_answer, ai_generated_answer):
        inputs = self.tokenizer(
            candidate_answer,
            ai_generated_answer,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="tf"  # Ensure TensorFlow tensors
        )
        outputs = self.model(inputs)
        score = outputs.logits.numpy().squeeze().item()  # Convert logits to numpy for processing
        return score

if __name__ == "__main__":
    fine_tune = CodeBERTFineTuner()
    file_path = "app/utils/results/features_with_aiscores.csv"

    # Load and preprocess data
    dataset = fine_tune.load_data(file_path)
    train_dataset, val_dataset = fine_tune.preprocess_data(dataset)

    # Train and save the model
    history = fine_tune.train(train_dataset, val_dataset)
    fine_tune.save_model()