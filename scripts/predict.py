#!/usr/bin/env python3
"""
Prediction Only Script - Uses pre-trained hybrid enhanced model
Loads existing model and makes predictions on test data
"""

import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from tqdm import tqdm
import gc

class HybridPredictor:
    def __init__(self, model_dir="model/hybrid_enhanced"):
        """Load pre-trained model components"""
        print("🔧 Loading pre-trained model components...")
        
        # Load TF-IDF vectorizer
        with open(f"{model_dir}/tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Load k-NN model
        with open(f"{model_dir}/knn_model.pkl", "rb") as f:
            self.knn_model = pickle.load(f)
        
        # Load training labels
        with open(f"{model_dir}/train_labels.pkl", "rb") as f:
            self.train_labels = pickle.load(f)
        
        # Turkish abbreviations dictionary
        self.abbreviations = {
            'mah.': 'mahallesi', 'mah': 'mahallesi', 'mh.': 'mahallesi', 'mh': 'mahallesi',
            'sk.': 'sokak', 'sk': 'sokak', 'sok.': 'sokak', 'sok': 'sokak',
            'cad.': 'caddesi', 'cad': 'caddesi', 'cd.': 'caddesi', 'cd': 'caddesi',
            'bul.': 'bulvarı', 'bul': 'bulvarı', 'blv.': 'bulvarı', 'blv': 'bulvarı',
            'kat.': 'kat', 'kat': 'kat', 'k.': 'kat',
            'd.': 'daire', 'd': 'daire',
            'apt.': 'apartmanı', 'apt': 'apartmanı',
            'no.': 'numara', 'no': 'numara', 'nr.': 'numara', 'nr': 'numara',
            'vs.': 've saire', 'vs': 've saire',
            'vb.': 've benzeri', 'vb': 've benzeri',
            'tr': 'türkiye', 'tr.': 'türkiye'
        }
        
        print("✅ Model components loaded successfully!")
        
    def normalize_address(self, text):
        """Normalize address text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Lowercase and apply unidecode for Turkish characters
        text = unidecode(text.lower())
        
        # Expand common abbreviations
        for abbrev, full in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, text)
        
        # Remove all punctuation except spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, query_texts, k=10):
        """Make predictions using pre-trained model"""
        print("🔮 Making predictions...")
        
        # Process queries in chunks
        chunk_size = 500
        all_predictions = []
        
        for i in tqdm(range(0, len(query_texts), chunk_size), desc="Predicting"):
            end_idx = min(i + chunk_size, len(query_texts))
            chunk_queries = query_texts[i:end_idx]
            
            # Normalize query texts
            normalized_queries = [self.normalize_address(text) for text in chunk_queries]
            
            # Transform to TF-IDF
            query_vectors = self.tfidf_vectorizer.transform(normalized_queries)
            
            # Find nearest neighbors
            distances, indices = self.knn_model.kneighbors(query_vectors)
            
            # Get predictions with weighted voting
            chunk_predictions = []
            for dist, idx in zip(distances, indices):
                neighbor_labels = [self.train_labels[j] for j in idx]
                
                # Weighted voting based on similarity
                weights = 1 / (dist + 1e-8)  # Avoid division by zero
                label_weights = {}
                
                for label, weight in zip(neighbor_labels, weights):
                    if label in label_weights:
                        label_weights[label] += weight
                    else:
                        label_weights[label] = weight
                
                # Get label with highest weight
                predicted_label = max(label_weights.items(), key=lambda x: x[1])[0]
                chunk_predictions.append(predicted_label)
            
            all_predictions.extend(chunk_predictions)
            
            # Force garbage collection
            del query_vectors, distances, indices
            gc.collect()
        
        return all_predictions

def compare_submissions(old_file, new_file):
    """Compare two submission files and show differences"""
    print(f"\n📊 Comparing submissions...")
    
    if not os.path.exists(old_file):
        print(f"⚠️ Old submission file not found: {old_file}")
        return
    
    old_sub = pd.read_csv(old_file)
    new_sub = pd.read_csv(new_file)
    
    print(f"📈 Old submission: {len(old_sub):,} predictions")
    print(f"📈 New submission: {len(new_sub):,} predictions")
    
    # Count differences
    differences = (old_sub['label'] != new_sub['label']).sum()
    total = len(old_sub)
    difference_percentage = (differences / total) * 100
    
    print(f"🔄 Differences: {differences:,} out of {total:,} ({difference_percentage:.2f}%)")
    
    # Show some examples of differences
    if differences > 0:
        print(f"\n🔍 Sample differences:")
        diff_indices = old_sub[old_sub['label'] != new_sub['label']].index[:5]
        for idx in diff_indices:
            print(f"  ID {idx}: {old_sub.iloc[idx]['label']} → {new_sub.iloc[idx]['label']}")
    
    return differences, difference_percentage

def main():
    """Main prediction function"""
    print("🚀 Hybrid Enhanced Prediction Only")
    print("=" * 40)
    
    # Check if model exists
    model_dir = "model/hybrid_enhanced"
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        print("Please run the full pipeline first to train the model.")
        return
    
    # Load test data
    print("📂 Loading test data...")
    test_data = pd.read_csv("Data/testing.csv")
    print(f"📊 Test data loaded: {len(test_data):,} samples")
    
    # Initialize predictor
    predictor = HybridPredictor(model_dir)
    
    # Make predictions
    predictions = predictor.predict(test_data['address'].tolist(), k=10)
    
    # Create submission
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'label': predictions
    })
    
    # Save new submission
    new_submission_file = "Data/submission_new.csv"
    submission.to_csv(new_submission_file, index=False)
    
    print(f"\n✅ Predictions completed!")
    print(f"📊 Predictions: {len(predictions):,}")
    print(f"🏷️ Unique labels: {len(set(predictions)):,}")
    print(f"📁 New submission saved: {new_submission_file}")
    
    # Validate submission format
    expected_rows = 230101  # 0 to 230100
    if len(submission) == expected_rows:
        print("✅ Submission format correct")
    else:
        print(f"⚠️ Expected {expected_rows} rows, got {len(submission)}")
    
    # Show prediction distribution
    label_counts = submission['label'].value_counts()
    print(f"\n📊 Top 10 predicted labels:")
    print(label_counts.head(10))
    
    # Compare with existing submission if it exists
    old_submission_file = "Data/submission.csv"
    if os.path.exists(old_submission_file):
        differences, diff_percentage = compare_submissions(old_submission_file, new_submission_file)
        
        if differences == 0:
            print(f"\n🎉 No differences found! Predictions are identical.")
        else:
            print(f"\n📈 {differences:,} predictions changed ({diff_percentage:.2f}%)")
    else:
        print(f"\n📝 No existing submission found for comparison")

if __name__ == "__main__":
    main()
