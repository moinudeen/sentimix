# Sentimix Challenge
- The task is to predict the sentiment of a given code-mixed tweet. The sentiment labels are positive, negative, or neutral, and the code-mixed languages will be English-Hindi.
# Contents
This repo will cover the following things in their corresponding notebooks:
- 1. Data Exploration and Visualization
- 2. Classic ML models for baseline
- 3. Transformer based Deep Learning models 
- 4. Testset Evaluation Reports

## Installation

Clone this repo:

```
git clone https://github.com/moinudeen/sentimix.git
cd sentimix
```

Install the dependencies:

```
pip install -r requirements.txt
```


To deploy the trained model in an app, please follow the steps below:
```
cd src/
uvicorn api.main:app --reload
```

The model is deployed by using FastAPI and uvicorn. Go to `http://127.0.0.1:8000/docs` to see api documentation. 
![API Docs](docs.png)
For Simplicity, only the Logistic regression model has been uploaded to this repo. 
You can update the api endpoint to point to your own trained model by changing the path value in `src/api/model_registry.json`
