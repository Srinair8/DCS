#!/usr/bin/env python3
# quick_fix.py - Quick fix for the DistilBERT configuration issue

import json
import os
import sys
import shutil
from pathlib import Path

def find_model_directory():
    """Find the model directory in common locations"""
    possible_paths = [
        "drug_classifier_model",
        "./drug_classifier_model", 
        "model",
        "./model",
        "../drug_classifier_model",
        sys.argv[1] if len(sys.argv) > 1 else None
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path) and os.path.isdir(path):
            return path
    
    return None

def fix_config_file(model_path):
    """Fix the configuration file"""
    config_path = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Backup original
    backup_path = config_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)
        print(f"✅ Backed up config to: {backup_path}")
    
    try:
        # Read config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Original config: dim={config.get('dim')}, n_heads={config.get('n_heads')}")
        
        # Fix the problematic values
        config['dim'] = 768
        config['n_heads'] = 12
        config['hidden_dim'] = 3072
        
        # Ensure other required fields
        config.update({
            'vocab_size': 30522,
            'max_position_embeddings': 512,
            'n_layers': 6,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'activation': 'gelu',
            'initializer_range': 0.02,
            'qa_dropout': 0.1,
            'seq_classif_dropout': 0.2,
            'num_labels': 2
        })
        
        # Write fixed config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Fixed config: dim={config['dim']}, n_heads={config['n_heads']}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing config: {e}")
        return False

def main():
    print("🔧 DistilBERT Configuration Fix Script")
    print("="*40)
    
    # Find model directory
    model_path = find_model_directory()
    
    if not model_path:
        print("❌ Model directory not found!")
        print("Usage: python quick_fix.py [model_path]")
        print("Or place this script in the same directory as your model folder")
        return
    
    print(f"📁 Found model at: {model_path}")
    
    # Check current config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            dim = config.get('dim', 768)
            n_heads = config.get('n_heads', 12)
            
            print(f"📋 Current config: dim={dim}, n_heads={n_heads}")
            
            if dim % n_heads == 0:
                print("✅ Configuration is already valid!")
                return
            else:
                print(f"❌ Problem detected: {dim} % {n_heads} = {dim % n_heads} (should be 0)")
                
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    
    # Fix the config
    print("\n🔧 Applying fix...")
    success = fix_config_file(model_path)
    
    if success:
        print("\n🎉 Fix applied successfully!")
        print("You can now run your application without the configuration error.")
        
        # Verify fix
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Verified: dim={config['dim']}, n_heads={config['n_heads']}")
        except:
            print("⚠️  Could not verify fix")
            
    else:
        print("\n❌ Fix failed!")
        print("You may need to retrain your model or check the backup file.")

if __name__ == "__main__":
    main()