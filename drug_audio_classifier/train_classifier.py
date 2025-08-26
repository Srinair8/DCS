import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We'll now primarily use your existing 1500-entry dataset
# Synthetic examples are only used for targeted augmentation if needed

def load_and_augment_data(csv_path="train.csv"):
    """Load your existing 1500-entry dataset with optional targeted augmentation"""
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data file not found: {csv_path}")
    
    # Load your existing dataset
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} examples from {csv_path}")
    
    # Validate columns
    if not {'text', 'label'}.issubset(df.columns):
        raise ValueError("train.csv must contain 'text' and 'label' columns")
    
    # Check for invalid or missing data
    initial_len = len(df)
    df = df.dropna(subset=['text', 'label'])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing data")
    
    # Clean and validate labels - handle various possible formats
    df['label'] = df['label'].astype(str).str.upper().str.strip()
    
    # Map common label variations to standard format
    label_mapping = {
        'DRUG': 'DRUG',
        'NON_DRUG': 'NON_DRUG',
        'NON-DRUG': 'NON_DRUG',
        'NONDRUG': 'NON_DRUG',
        'NOT_DRUG': 'NON_DRUG',
        'NO_DRUG': 'NON_DRUG',
        '1': 'DRUG',
        '0': 'NON_DRUG',
        'TRUE': 'DRUG',
        'FALSE': 'NON_DRUG',
        'YES': 'DRUG',
        'NO': 'NON_DRUG'
    }
    
    # Apply label mapping
    df['label'] = df['label'].map(label_mapping).fillna(df['label'])
    
    # Check for any remaining invalid labels
    valid_labels = ['DRUG', 'NON_DRUG']
    invalid_mask = ~df['label'].isin(valid_labels)
    
    if invalid_mask.any():
        invalid_labels = df.loc[invalid_mask, 'label'].unique()
        logger.warning(f"Found {invalid_mask.sum()} rows with invalid labels: {invalid_labels}")
        logger.warning("These will be dropped. Valid labels are: DRUG, NON_DRUG")
        df = df[~invalid_mask]
    
    # Analyze your dataset balance
    label_counts = df['label'].value_counts()
    drug_count = label_counts.get("DRUG", 0)
    non_drug_count = label_counts.get("NON_DRUG", 0)
    drug_ratio = drug_count / len(df) if len(df) > 0 else 0
    
    logger.info(f"Your dataset analysis:")
    logger.info(f"  Total examples: {len(df)}")
    logger.info(f"  DRUG examples: {drug_count} ({drug_ratio:.1%})")
    logger.info(f"  NON_DRUG examples: {non_drug_count} ({(1-drug_ratio):.1%})")
    
    # Check if we need targeted augmentation
    need_augmentation = False
    augmentation_reason = []
    
    if drug_ratio < 0.2:  # Less than 20% drug examples
        need_augmentation = True
        augmentation_reason.append(f"low DRUG ratio ({drug_ratio:.1%})")
    
    if drug_count < 100:  # Less than 100 drug examples
        need_augmentation = True
        augmentation_reason.append(f"low DRUG count ({drug_count})")
    
    # Optional targeted augmentation for specific missing patterns
    if need_augmentation:
        logger.info(f"Dataset needs augmentation due to: {', '.join(augmentation_reason)}")
        logger.info("Adding targeted synthetic examples to improve model robustness...")
        
        # Add only the most critical synthetic examples that might be missing
        critical_drug_examples = [
            {"text": "Bro, check the Insta DM. That the white or the blue?", "label": "DRUG"},
            {"text": "White, straight from Mumbai. Cool, payment through crypto, right?", "label": "DRUG"},
            {"text": "Who's bringing the stuff? Raj, Tabs, Weed and Coke.", "label": "DRUG"},
            {"text": "Let's not overdose this time.", "label": "DRUG"},
            {"text": "Saturday Rave is confirmed, right? Yes, outskirts near Kanaka Pura.", "label": "DRUG"},
            {"text": "Got the hash and charas ready for pickup.", "label": "DRUG"},
            {"text": "Quality MDMA and LSD tabs available.", "label": "DRUG"},
            {"text": "Syringe and needle for the gear.", "label": "DRUG"},
            {"text": "Trip was amazing, need more powder.", "label": "DRUG"},
            {"text": "Package delivery confirmed, bring crypto payment.", "label": "DRUG"},
        ]
        
        synthetic_df = pd.DataFrame(critical_drug_examples)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        
        logger.info(f"Added {len(critical_drug_examples)} targeted synthetic DRUG examples")
    else:
        logger.info("Dataset appears well-balanced, using your original data without augmentation")
    
    # Final statistics
    final_counts = df['label'].value_counts()
    final_drug_ratio = final_counts.get("DRUG", 0) / len(df)
    
    logger.info(f"Final dataset: {len(df)} examples")
    logger.info(f"Final DRUG ratio: {final_drug_ratio:.1%} ({final_counts.get('DRUG', 0)} examples)")
    logger.info(f"Final NON_DRUG ratio: {(1-final_drug_ratio):.1%} ({final_counts.get('NON_DRUG', 0)} examples)")
    
    return df

# Custom weighted loss for class imbalance
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weight_tensor = torch.tensor(self.class_weights, device=labels.device, dtype=torch.float)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def main():
    # Load and prepare data
    df = load_and_augment_data()
    
    # Encode labels
    label2id = {"NON_DRUG": 0, "DRUG": 1}
    id2label = {0: "NON_DRUG", 1: "DRUG"}
    df['label_id'] = df['label'].map(label2id)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label_id']),
        y=df['label_id']
    )
    logger.info(f"Computed class weights: NON_DRUG={class_weights[0]:.3f}, DRUG={class_weights[1]:.3f}")
    
    # Split dataset with stratification
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label_id'].tolist(), 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_id']  # Ensure balanced split
    )
    
    logger.info(f"Training set: {len(train_texts)} samples")
    logger.info(f"Validation set: {len(val_texts)} samples")
    
    # Check balance in splits
    train_drug_ratio = sum(train_labels) / len(train_labels)
    val_drug_ratio = sum(val_labels) / len(val_labels)
    logger.info(f"Train DRUG ratio: {train_drug_ratio:.2%}")
    logger.info(f"Validation DRUG ratio: {val_drug_ratio:.2%}")
    
    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Tokenize with appropriate max length
    max_length = 256  # Reduced for efficiency, most drug conversations are short
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    
    # Dataset class
    class DrugDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = DrugDataset(train_encodings, train_labels)
    val_dataset = DrugDataset(val_encodings, val_labels)
    
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    
    # Enhanced training arguments optimized for your 1500-entry dataset
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=6,  # Good balance for 1500 examples
        per_device_train_batch_size=8,  # Suitable for most GPUs
        per_device_eval_batch_size=16,  # Larger batch for evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,  # Log every 10 steps
        learning_rate=2e-5,  # Standard DistilBERT learning rate
        weight_decay=0.01,  # L2 regularization
        warmup_steps=len(train_dataset) // 10,  # 10% of training steps
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        seed=42,  # For reproducibility
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_drop_last=False,
        report_to=None,  # Disable wandb/tensorboard logging
    )
    
    # Enhanced metrics computation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        
        # Per-class metrics
        logger.info(f"Eval metrics:")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  NON_DRUG - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
        logger.info(f"  DRUG - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
        logger.info(f"  Macro avg - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        
        # Classification report
        logger.info("Detailed classification report:")
        logger.info(f"\n{classification_report(labels, preds, target_names=['NON_DRUG', 'DRUG'])}")
        
        return {
            'accuracy': acc,
            'f1': f1_macro,  # Use macro F1 as main metric
            'f1_drug': f1[1],  # F1 for DRUG class specifically
            'precision': precision_macro,
            'recall': recall_macro,
            'precision_drug': precision[1],
            'recall_drug': recall[1],
        }
    
    # Use weighted trainer to handle class imbalance
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Save model and tokenizer
    output_dir = "drug_classifier_model"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to '{output_dir}'")
    
    # Test with sample drug-related text
    logger.info("Testing model with sample drug-related text...")
    test_text = "Bro, check the Insta DM. That the white or the blue? White, straight from Mumbai. Cool, payment through crypto, right? Who's bringing the stuff? Raj, Tabs, Weed and Coke. Let's not overdose this time."
    
    # Tokenize and predict
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        drug_probability = predictions[0][1].item()  # Probability of DRUG class
    
    logger.info(f"Test prediction: {'DRUG' if predicted_class == 1 else 'NON_DRUG'}")
    logger.info(f"DRUG probability: {drug_probability:.4f} ({drug_probability*100:.2f}%)")
    logger.info(f"NON_DRUG probability: {1-drug_probability:.4f} ({(1-drug_probability)*100:.2f}%)")

if __name__ == "__main__":
    main()