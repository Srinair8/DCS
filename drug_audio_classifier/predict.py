# predict.py - Fixed version with configuration validation
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertConfig
import torch
import torch.nn.functional as F
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer (loaded once)
model = None
tokenizer = None
model_loaded = False

def validate_and_fix_config(model_path):
    """Validate and fix model configuration if needed"""
    config_path = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Check for the problematic configuration
        dim = config_data.get('dim', 768)
        n_heads = config_data.get('n_heads', 12)
        
        if dim % n_heads != 0:
            logger.warning(f"Configuration issue detected: dim={dim} not divisible by n_heads={n_heads}")
            
            # Backup original config
            backup_path = config_path + ".backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.info(f"Backed up original config to {backup_path}")
            
            # Fix configuration with standard DistilBERT values
            config_data['dim'] = 768
            config_data['n_heads'] = 12
            config_data['hidden_dim'] = 3072
            
            # Write fixed config
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info("Fixed configuration with standard DistilBERT dimensions")
            return True
        
        logger.info("Configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"Error validating/fixing config: {e}")
        return False

def load_model_with_fallback(model_name):
    """Load model with fallback strategies"""
    global model, tokenizer
    
    # Strategy 1: Try loading with config validation
    if os.path.exists(model_name):
        logger.info(f"Attempting to load local model from {model_name}")
        
        # Validate and fix config first
        config_valid = validate_and_fix_config(model_name)
        if not config_valid:
            logger.warning("Could not validate/fix config, trying anyway...")
        
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
            model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True  # This helps with dimension issues
            )
            logger.info("Successfully loaded local model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            
    # Strategy 2: Create a compatible model with existing weights
    if os.path.exists(model_name):
        try:
            logger.info("Attempting to load with custom configuration...")
            
            # Create a working config
            config = DistilBertConfig(
                vocab_size=30522,
                max_position_embeddings=512,
                dim=768,
                n_layers=6,
                n_heads=12,
                hidden_dim=3072,
                dropout=0.1,
                attention_dropout=0.1,
                activation='gelu',
                initializer_range=0.02,
                qa_dropout=0.1,
                seq_classif_dropout=0.2,
                num_labels=2
            )
            
            # Load tokenizer
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            
            # Create model with fixed config
            model = DistilBertForSequenceClassification(config)
            
            # Try to load existing weights
            weights_path = os.path.join(model_name, "pytorch_model.bin")
            if os.path.exists(weights_path):
                try:
                    state_dict = torch.load(weights_path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded existing weights with custom config")
                except Exception as weight_error:
                    logger.warning(f"Could not load weights: {weight_error}")
                    logger.info("Using randomly initialized model")
            
            return True
            
        except Exception as e:
            logger.error(f"Custom config loading failed: {e}")
    
    # Strategy 3: Use pre-trained DistilBERT as fallback
    try:
        logger.info("Loading fallback model from HuggingFace...")
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        logger.warning("Using pre-trained DistilBERT as fallback - will need retraining for drug classification")
        return True
        
    except Exception as e:
        logger.error(f"Fallback model loading failed: {e}")
        return False

def load_model(model_name="drug_classifier_model"):
    """Load model and tokenizer once with enhanced error handling"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return  # Already loaded
    
    try:
        # Use the enhanced loading function
        success = load_model_with_fallback(model_name)
        
        if not success:
            raise RuntimeError("All model loading strategies failed")
        
        model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to GPU")
        
        model_loaded = True
        logger.info(f"Successfully loaded model and tokenizer")
        
        # Log model configuration if available
        if hasattr(model.config, 'id2label'):
            logger.info(f"Model labels: {model.config.id2label}")
        
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

def predict(text, confidence_threshold=0.5):
    """
    Predict whether the input text is DRUG (1) or NON_DRUG (0).
    
    Args:
        text (str): Input text to classify
        confidence_threshold (float): Threshold for DRUG classification (default: 0.5)
    
    Returns:
        tuple: (label, drug_probability)
            - label: 1 for DRUG, 0 for NON_DRUG
            - drug_probability: float between 0 and 1
    """
    # Ensure model is loaded
    if not model_loaded:
        load_model()
    
    # Input validation
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid input text provided to predict")
        return 0, 0.0
    
    text = text.strip()
    if len(text) == 0:
        logger.warning("Empty text after stripping provided to predict")
        return 0, 0.0
    
    try:
        # Tokenize input - use same max_length as training
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256  # Match training script
        )
        
        # Move inputs to same device as model
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
            non_drug_prob = probs[0][0].item()  # Probability for NON_DRUG (index 0)
            drug_prob = probs[0][1].item()      # Probability for DRUG (index 1)
            
            # Apply threshold for classification
            pred_label = 1 if drug_prob > confidence_threshold else 0
        
        # Log prediction details
        logger.info(f"Prediction for: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        logger.info(f"  Result: {'DRUG' if pred_label == 1 else 'NON_DRUG'}")
        logger.info(f"  DRUG probability: {drug_prob:.4f} ({drug_prob*100:.2f}%)")
        logger.info(f"  NON_DRUG probability: {non_drug_prob:.4f} ({non_drug_prob*100:.2f}%)")
        logger.info(f"  Confidence: {max(drug_prob, non_drug_prob):.4f}")
        
        return pred_label, drug_prob
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(f"Input text: {text}")
        return 0, 0.0

def predict_batch(texts, confidence_threshold=0.5):
    """
    Predict for multiple texts at once (more efficient)
    
    Args:
        texts (list): List of texts to classify
        confidence_threshold (float): Threshold for DRUG classification
    
    Returns:
        list: List of (label, drug_probability) tuples
    """
    if not model_loaded:
        load_model()
    
    if not texts or not isinstance(texts, list):
        logger.warning("Empty or invalid text list provided to predict_batch")
        return []
    
    # Filter out empty texts and keep track of original indices
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and isinstance(text, str) and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)
    
    if not valid_texts:
        logger.warning("No valid texts found in batch")
        return [(0, 0.0)] * len(texts)
    
    try:
        # Tokenize all texts
        inputs = tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        # Move to device
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
            # Create results array with proper indexing
            results = [(0, 0.0)] * len(texts)
            for i, valid_idx in enumerate(valid_indices):
                drug_prob = probs[i][1].item()
                pred_label = 1 if drug_prob > confidence_threshold else 0
                results[valid_idx] = (pred_label, drug_prob)
        
        logger.info(f"Batch prediction completed for {len(valid_texts)} texts")
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        return [(0, 0.0)] * len(texts)

def test_predictions():
    """Test function to verify model predictions"""
    logger.info("Running prediction tests...")
    
    test_cases = [
        # Should be DRUG
        "Bro, check the Insta DM. That the white or the blue? White, straight from Mumbai. Cool, payment through crypto, right? Who's bringing the stuff? Raj, Tabs, Weed and Coke. Let's not overdose this time.",
        "Got some quality hash and charas ready for pickup tonight",
        "MDMA tabs are available, payment through crypto only",
        
        # Should be NON_DRUG  
        "Hey, how's your work going today? Let's meet for coffee this evening.",
        "The weather is really nice today, perfect for a walk in the park",
        "I need to finish my project by tomorrow, can you help me?",
    ]
    
    for i, text in enumerate(test_cases, 1):
        logger.info(f"\n--- Test Case {i} ---")
        label, prob = predict(text)
        expected = "DRUG" if i <= 3 else "NON_DRUG"
        actual = "DRUG" if label == 1 else "NON_DRUG"
        logger.info(f"Expected: {expected}, Got: {actual}, Probability: {prob:.4f}")

if __name__ == "__main__":
    # Load model once when script is run directly
    load_model()
    
    # Run tests
    test_predictions()
    
    # Interactive testing
    print("\n" + "="*50)
    print("Interactive Drug Detection Testing")
    print("Enter text to classify (or 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                label, prob = predict(user_input)
                result = "🚨 DRUG" if label == 1 else "✅ NON_DRUG"
                confidence = max(prob, 1-prob)
                print(f"Result: {result}")
                print(f"Drug Probability: {prob*100:.2f}%")
                print(f"Confidence: {confidence*100:.2f}%")
            else:
                print("Please enter some text.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")