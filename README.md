# AI Plagiarism Detection using LightGBM and CodeBERT
## Streamlit / Usage Overview 
This project provides a user-friendly Streamlit interface for evaluating similarity metrics between AI-generated and candidate answers using **LightGBM** and **CodeBERT**. Here's a preview of the 

![alt text](https://github.com/nsyawali12/test_case_candidate_ai/blob/main/screenshot_streamlit.png)

## Getting Started

### Prerequisites

Ensure you have the following installed:
- **Anaconda** (for managing environments)
- **Python 3.10.16**
- **Streamlit** and **FastAPI** (required to install `requirements.txt` should be okay)
    ```requirement.txt
    pandas
    matplotlib
    seaborn
    numpy
    scikit-learn
    transformers
    nltk
    spacy
    torch
    torchvision
    torchaudio
    tensorflow
    streamlit
    fastapi
    uvicorn
    jupyter
    openpyxl
    python-Levenshtein
    lightgbm
    tf-keras
    ```
    

### Setup Guide
1.  **Set Up the Conda Environment**: Create and activate a Conda environment with all necessary dependencies.
2. Anaconda Environtment and istall dependencies
   ```
   conda create -n freename_env python=3.10.16
   conda activate freename_env
   pip install -r requirements.txt
   ```

4. Run Fast API Server in Terminal or Powershell Conda
      
    ```
    uvicorn app.main:app --host 127.0.0.1 --port 8080
    ```
5. **Run the Streamlit App**: In a new terminal, launch the Streamlit app.
   ```bash
   streamlit run streamlit_app.py
   ```
## Working Roadmap with Phase
![alt text](https://github.com/nsyawali12/test_case_candidate_ai/blob/main/roadmap_or_planning.png)

1. Phase 1: Understanding the Data
![alt_text](https://github.com/nsyawali12/test_case_candidate_ai/blob/main/data_understanding.png)
To understand the data, we compared source codes with various GPT-generated models. Each comparison was stored as metadata, including scores for plagiarism detection. Scores near 0 indicate human-like content, while scores closer to 1 suggest AI-generated content. This process helping me on the analysis of the data it self and for further processing.

2. Phase 2: Preprocessing and Feature Engineering


   
