from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from nlp import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, pipeline


class SentimixDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        # print(item)
        return item

    def __len__(self):
        return len(self.labels)


class SentimixTransformer:
    """
    HuggingFace Transformer based classifier for Sentimix.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = None
        self.inference_pipeline = None

    def preprocess_data(
        self, data_path: str, text_col: str = None, target_col: str = None
    ) -> Tuple[Tuple[SentimixDataset], List[str]]:
        """
        preprocess the raw data, tokenize and split train/val, generate the dataset.
        """
        df = pd.read_csv(data_path)
        train_df = df.sample(frac=0.9, random_state=200)
        val_df = df.drop(train_df.index)
        labels = sorted(df["sentiment"].unique().tolist())
        train_texts, train_labels = (
            train_df[text_col].tolist(),
            train_df[target_col].tolist(),
        )
        test_texts, test_labels = val_df[text_col].tolist(), val_df[target_col].tolist()

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)

        train_dataset = SentimixDataset(train_encodings, train_labels)
        test_dataset = SentimixDataset(test_encodings, test_labels)

        return train_dataset, test_dataset, labels

    def train(self, training_args, train_dataset) -> None:
        """
        Start training process.
        """
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
        )

        self.trainer.train()

    def evaluate(self, test_dataset: SentimixDataset) -> dict:
        """
        Evaluate the model.
        """
        return self.trainer.evaluate(eval_dataset=test_dataset)

    def build_pipeline(self):
        """
        Build text classification pipeline with this model.
        """
        pipe = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0
        )
        self.inference_pipeline = pipe

    def save_model(self, model_path: str) -> None:
        """
        save the model and tokenizer.
        """
        self.trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)

    def compute_metrics(self, pred):
        """
        compute metrics for eval.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
