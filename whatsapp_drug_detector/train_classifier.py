from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import torch
import os
import numpy as np
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_validate_data(csv_path="data/conversations.csv"):
    """Load and validate the training data"""
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        # Validate required columns
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        
        # Clean and validate data
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str).str.strip()
        df['label'] = df['label'].astype(int)
        
        # Validate labels are binary (0 or 1)
        unique_labels = df['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError("Labels must be binary (0 for clean, 1 for suspicious)")
        
        # Check class distribution
        class_counts = df['label'].value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            raise ValueError("Dataset must contain both classes (0 and 1)")
        
        # Warn about class imbalance
        minority_ratio = min(class_counts) / len(df)
        if minority_ratio < 0.1:
            logger.warning(f"Severe class imbalance detected: {minority_ratio:.2%} minority class")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def tokenize_function(examples, tokenizer):
    """Tokenize the text data and preserve labels"""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",   # ✅ fixed padding for consistency
        truncation=True,
        max_length=128,         # ✅ enough for short chat messages
        return_tensors=None
    )
    # Important: Add labels to the tokenized output
    tokenized["labels"] = examples["label"]
    return tokenized

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist()
    }

def create_model_info(df, model_path):
    """Create model information file"""
    info = {
        "model_type": "DistilBERT for Drug Detection",
        "training_date": pd.Timestamp.now().isoformat(),
        "training_samples": len(df),
        "class_distribution": df['label'].value_counts().to_dict(),
        "model_path": model_path,
        "tokenizer": "distilbert-base-uncased"
    }
    
    with open(os.path.join(model_path, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

def main():
    try:
        # Step 1: Load and validate data
        logger.info("Loading and validating training data...")
        df = load_and_validate_data("data/conversations.csv")
        
        # Step 2: Create dataset
        logger.info("Creating dataset...")
        dataset = Dataset.from_pandas(df)
        
        # Step 3: Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        
        # Step 4: Tokenize data
        logger.info("Tokenizing data...")
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=["text"]  # Remove text column but keep label as labels
        )
        
        # Step 5: Train-validation-test split
        logger.info("Splitting data...")
        # First split: 80% train+val, 20% test
        train_val_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        # Second split: 80% train, 20% val from the train+val portion
        train_val = train_val_test['train'].train_test_split(test_size=0.25, seed=42)  # 0.25 * 0.8 = 0.2 overall
        
        dataset_splits = DatasetDict({
            'train': train_val['train'],
            'validation': train_val['test'],
            'test': train_val_test['test']
        })
        
        logger.info(f"Dataset splits - Train: {len(dataset_splits['train'])}, "
                   f"Validation: {len(dataset_splits['validation'])}, "
                   f"Test: {len(dataset_splits['test'])}")
        
        # Step 6: Load model
        logger.info("Loading DistilBERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2
        )
        
        # Step 7: Create output directory
        model_dir = "model/saved_model"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Step 8: Setup training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            push_to_hub=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=True,
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Step 9: Setup trainer with early stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_splits["train"],
            eval_dataset=dataset_splits["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Step 10: Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Step 11: Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=dataset_splits["test"])
        logger.info(f"Test Results: {test_results}")
        
        # Step 12: Save model and tokenizer
        logger.info("Saving model and tokenizer...")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Step 13: Save training info
        create_model_info(df, model_dir)
        
        # Save test results
        with open(os.path.join(model_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"Model saved to: {model_dir}")
        logger.info(f"Test F1 Score: {test_results.get('eval_f1', 'N/A'):.4f}")
        logger.info(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        
        return model_dir
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run training
    model_path = main()