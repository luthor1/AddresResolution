import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json
from tqdm import tqdm

train = pd.read_csv("Data/train.csv")
n_unique = train["label"].nunique()

class BertTrainer:
    def __init__(self, model_name = "dbmdz/bert-base-turkish-cased", num_labels = n_unique):
        print(f"Model ekleniyor: {model_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Kullanılan cihaz: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels
        )

        self.model.to(self.device)

        print("Model yüklendi")

         

def main():
    trainer = BertTrainer()
    
if __name__ == "__main__":
    main()

