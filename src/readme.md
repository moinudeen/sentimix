# Sentimix Challenge
- The task is to predict the sentiment of a given code-mixed tweet. The sentiment labels are positive, negative, or neutral, and the code-mixed languages will be English-Hindi.
# Contents
This repo will cover the following things in their corresponding notebook demos:
- 1. Data Exploration and Visualization
- 2. Classic ML models for baseline
- 3. Transformer based Deep Learning models 
- 4. Testset Evaluation Reports

## Modules
- `api` - code for deploying the trained model via FastAPI
    - Run the command `uvicorn api.main:app --reload` from the `src` directory.
- `models` - ML model code.
    - Used sklearn for classical ML architectures.
    - Used HuggingFace Transformers for Deep NLP Architectures.
- `notebooks` - Contains Dev Notebooks
    - Also contains a notebook where i tried to do Language Modelling from scratch. Didn't work so well as data was limited.
- `utils` - utility code for model training, data exploration ,etc.
