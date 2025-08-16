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

train_data = pd.read_csv("Data/train.csv")
n_unique = train_data["label"].nunique()

class BertTrainer:
    def __init__(self, model_name = "dbmdz/bert-base-turkish-cased", num_labels = n_unique):
        # Yapıcı modeli yükler ve cihazı ayarlar
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

    def load_data(self, train_file = "Data/train.csv", test_file = "Data/test.csv"):     

        # Verileri yükler
        print("Veriler yükleniyor...")
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        return self.train_data, self.test_data
    
    def prepare_batch(self, texts, labels):

        # Batch veriyi hazırlar
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return tokenized , labels_tensor
    
    def train_epoch(self, train_loader, optimizer, criterion):

        self.model.train()
        total_loss = 0

        for batch_texts, batch_labels in train_loader:

            #verileri cihaza taşı
            batch_texts = {k: v.to(self.device) for k, v in batch_texts.items()}
            batch_labels = batch_labels.to(self.device)
            
            optimizer.zero_grad()

            outputs = self.model(**batch_texts)
            loss = criterion(outputs.logits, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)
        
    def evaluate(self, test_loader, criterion):  
        #Modeli test eder  

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                # verileri cihaza taşı
                batch_texts = {k: v.to(self.device) for k, v in batch_texts.items()}
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(**batch_texts)
                loss = criterion(outputs.logits, batch_labels)

                predictions = torch.argmax(outputs.logits, dim=1)

                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

        return {
            "loss": total_loss / len(test_loader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(self, output_dir="model/fineModel", num_epochs=3, batch_size=32, learning_rate=2e-5):
        # Modeli eğitir
        print("Eğitim başlatılıyor...")
        print(f"Epoch sayısı: {num_epochs}")
        print(f"Batch boyutu: {batch_size}")
        print(f"Öğrenme oranı: {learning_rate}")

        os.makedirs(output_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        train_texts = self.train_data["address"].tolist()
        train_labels = self.train_data["label"].tolist()
        test_texts = self.test_data["address"].tolist()
        test_labels = self.test_data["label"].tolist()

        # Eğitim döngüsü

        best_f1 = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Eğitim verilerini batch'lere ayır
            train_batches = []
            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                tokenized, label_tensor = self.prepare_batch(batch_texts, batch_labels)
                train_batches.append((tokenized, label_tensor))

            # Test verilerini batch'lere ayır
            test_batches = []
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                batch_labels = test_labels[i:i + batch_size]
                tokenized, label_tensor = self.prepare_batch(batch_texts, batch_labels)
                test_batches.append((tokenized, label_tensor))

            # Eğitim
            train_loss = self.train_epoch(train_batches, optimizer, criterion)

            # Test
            test_results = self.evaluate(test_batches, criterion)

            print(f" Eğitim Kaybı: {train_loss:.4f}")
            print(f" Test Kaybı: {test_results['loss']:.4f}")
            print(f" Doğruluk: {test_results['accuracy']:.4f}")
            print(f" f1 Skoru: {test_results['f1']:.4f}")

            # En iyi modeli kaydet
            if test_results['f1'] > best_f1:
                best_f1 = test_results['f1']
                
                # Modeli kaydet
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                       
        # Model bilgilerini kaydet
        model_info = {
            "model_name": "BasicModel",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "test_accuracy": test_results['accuracy'],
            "test_f1": test_results['f1'],
            "test_precision": test_results['precision'],
            "test_recall": test_results['recall']
        }

        with open(f"{output_dir}/model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)

        print("Model eğitimi tamamlandı ve en iyi model kaydedildi.")
        print(f"En iyi f1 skoru: {best_f1:.4f}")

        return test_results
    

def main():

    print("Model eğitimi başlatılıyor...")
    print("-"* 50)

    # Veri kontrolü
    if not os.path.exists("Data/train.csv"):
        print("Eğitim verisi bulunamadı.")
        return
    
    # Trainer oluşturma
    trainer = BertTrainer()

    # Verileri yükleme
    train_data, test_data = trainer.load_data()

    # Modeli eğitme
    result = trainer.train(
        num_epochs=5,
        batch_size=1,
        learning_rate=2e-5,
    )

    print("\nFinal test sonuçları:")
    print(f"Doğruluk: {result['accuracy']:.2%}")
    print(f"F1 Skoru: {result['f1']:.2%}")
    print(f"Kesinlik: {result['precision']:.2%}")
    print(f"Hatırlama: {result['recall']:.2%}")

    
if __name__ == "__main__":
    main()

