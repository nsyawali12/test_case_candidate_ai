import tensorflow as tf

def initialize_tokenizer():
    """Initialize and return a TensorFlow tokenizer."""
    return tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")

def tokenize_texts(tokenizer, candidate_answer, ai_generated_answer):
    """Tokenize candidate and AI-generated answers."""
    candidate_seq = tokenizer.texts_to_sequences([candidate_answer])[0]
    ai_seq = tokenizer.texts_to_sequences([ai_generated_answer])[0]
    return candidate_seq, ai_seq


if __name__ == "__main__":
    # Example usage
    tokenizer = initialize_tokenizer()
    tokenizer.fit_on_texts(["This is an example candidate answer", "This is an AI-generated answer"])
    
    candidate = "This is an example candidate answer"
    ai_generated = "This is an AI-generated answer"
    
    candidate_seq, ai_seq = tokenize_texts(tokenizer, candidate, ai_generated)
    print("Candidate Sequence:", candidate_seq)
    print("AI-Generated Sequence:", ai_seq)