import pandas as pd
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF

from asin_utils import (
    standardize_brand, 
    get_keyword_mapping, 
    get_brand_subbrand_mapping,
    clean_text,
    map_subcategory_with_keywords,
    normalize_text,
    get_silver_span
)

# Import the map_subcategory_with_keywords function from asin_utils
from asin_utils import map_subcategory_with_keywords

#------------CRF FUNCTIONS FOR SIZE EXTRACTION-------------------------------------
#----------------------------------------------------------------------------------
def tokenize(text: str):
    """Split into words (with dot) or any single char (so hyphens separate)."""
    return re.findall(r"\w+\.?\w*|\S", text)

def silver_labels(tokens, span):
    """Map the silver span onto tokens using IOB tags."""
    labels = ["O"] * len(tokens)
    if not span:
        return labels

    # tokenize the span the same way we tokenize titles
    span_toks = [t for t in tokenize(span) if re.match(r"[A-Za-z0-9\.]+", t)]
    L = len(span_toks)

    for i in range(len(tokens) - L + 1):
        window = [t.lower().rstrip(".") for t in tokens[i:i+L]]
        target = [t.lower().rstrip(".") for t in span_toks]
        if window == target:
            labels[i] = "B-SIZE"
            for j in range(1, L):
                labels[i+j] = "I-SIZE"
            break

    return labels

# mirror the regex‐units you used for silver spans
UNIT_TOKENS = {
    "fl oz","floz","oz","ounce","ounces",
    "ml","milliliter","milliliters","millilitre","millilitres",
    "l","liter","litre","liters","litres",
    "g","gram","grams","kg","mg",
    "ct","count","ea","item","items","pair","pairs", "pcs",
    "pack","packs","packet","packets","set","sets"
}

def token2features(tokens, i):
    tok = tokens[i].lower().rstrip(".")
    
    # Basic features
    features = {
        "bias": 1.0,
        "word": tok,
        "has_digit": bool(re.search(r"\d", tok)),
        "unit_like": tok in UNIT_TOKENS,
        "prev_word": tokens[i-1].lower().rstrip(".") if i>0 else "",
        "next_word": tokens[i+1].lower().rstrip(".") if i< len(tokens)-1 else "",
    }
    
    # Add position features
    features["is_last_token"] = (i == len(tokens) - 1)
    features["is_in_parenthesis"] = False
    
    # Check if token is inside parentheses
    open_count = 0
    for j in range(i):
        if tokens[j] == '(':
            open_count += 1
        elif tokens[j] == ')':
            open_count -= 1
    if open_count > 0:
        features["is_in_parenthesis"] = True
    
    # Check if token is part of "X-in-Y" pattern (like "2-in-1")
    if i > 1 and i < len(tokens) - 1:
        if (tokens[i-1] == '-' and tokens[i] == 'in' and tokens[i+1] == '-') or \
           (re.match(r'\d+', tokens[i-2]) and tokens[i-1] == '-' and tokens[i] == 'in'):
            features["is_in_x_in_y_pattern"] = True
        else:
            features["is_in_x_in_y_pattern"] = False
    
    # Proximity to end of string (sizes often come at the end)
    features["rel_position"] = i / len(tokens)
    
    return features

def train_crf_size_extractor(df):
    """Train CRF model for size extraction"""
    print("Training CRF model for size extraction...")
    
    X_silver = []
    y_silver = []
    
    for name in df["item_name"]:
        # 1) normalize
        norm = normalize_text(name)
        # 2) tokenize normalized text
        toks = tokenize(norm)
        # 3) get silver span (normalized too)
        span = get_silver_span(name)  # this already calls normalize_text internally
        # 4) build features on toks, and silver labels on toks
        X_silver.append([token2features(toks, i) for i in range(len(toks))])
        y_silver.append(silver_labels(toks, span))
    
    crf = CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    # Train CRF
    crf.fit(X_silver, y_silver)
    print("✅ CRF training complete")
    
    return crf

#------------ ML FUNCTIONS FOR SUB-BRAND CLASSIFICATION-------------------------------------
#-------------------------------------------------------------------------------------------

def train_ml_subbrand_classifier(df, sample_size=None):  # Changed: sample_size=None (no limit)
    """Train ML classifier to map original_sub_brand to standard sub_brand"""
    print("Training ML classifier for sub-brand mapping...")
    
    if sample_size is None:
        training_df = df.copy()
        print(f"Using full dataset: {len(training_df):,} samples")
    else:
        sample_size = min(sample_size, len(df))
        training_df = df.head(sample_size).copy()
        print(f"Using limited dataset: {len(training_df):,} samples")
    
    # Apply keyword mapping
    training_df['mapped_sub_brand'] = training_df.apply(map_subcategory_with_keywords, axis=1)
    
    # Create features by combining original_sub_brand and item_name
    training_df['feature_text'] = training_df.apply(
        lambda row: f"{clean_text(row['original_sub_brand'])} {clean_text(row['item_name'])}",
        axis=1
    )
    
    # Train models by brand
    models = {}
    vectorizers = {}
    
    for brand_name, group in training_df.groupby('brand_name'):
       
        brand = str(brand_name).lower()
        print(f"Training model for brand: {brand} with {len(group)} samples")
        
        # Get texts and labels
        texts = group['feature_text'].values
        labels = group['mapped_sub_brand'].values
        
        # If only one class, store the single class
        if len(set(labels)) == 1:
            models[brand] = labels[0]
            print(f"  Single category brand - storing: {labels[0]}")
            continue
        
        # For brands with very few samples, still try to train if multiple classes
        min_samples_needed = max(2, len(set(labels)))  # At least 2 samples, or 1 per class
        
        if len(texts) < min_samples_needed:
            models[brand] = labels[0] if labels.size > 0 else ""
            print(f"  Too few samples ({len(texts)}) for training - using fallback")
            continue
            
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            min_df=1,  
            max_features=10000 
        )
        
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError as e:
            print(f"  Error creating features for {brand}: {e}")
            models[brand] = labels[0] if labels.size > 0 else ""
            continue
        
        # For smaller data set
        if len(texts) < 20:
            print(f"  Small dataset - training on all {len(texts)} samples")
            clf = RandomForestClassifier(
                n_estimators=50,  
                random_state=42,
                min_samples_split=2,
                min_samples_leaf=1
            )
            clf.fit(X, labels)
            
            y_pred = clf.predict(X)
            
        else:
            # Normal train/test split for larger datasets
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=42, stratify=labels
                )
            except ValueError:  
                # If stratification fails (not enough samples per class), do random split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=42
                )
            
            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)

        
        # Store model and vectorizer
        models[brand] = clf
        vectorizers[brand] = vectorizer
        print(f"ML model trained for {brand}") 
        
    return models, vectorizers
def main():
    """Main training function - trains both CRF and ML models"""
    print("=== COMPLETE MODEL TRAINING ===")
    print("This will train:")
    print("1. CRF model for size extraction")
    print("2. ML models for sub-brand classification")
    print()
    
    #---------------------------------------------------------------------------------
    # Load data directly from CSV file
    print("Loading data from all_asins.csv...")
    df = pd.read_csv(r"C:\Users\AmosXiao\Desktop\all_asins.csv", dtype=str).fillna("")
    
    # Process the data to match what training expects
    df['original_sub_brand'] = (
        df['Subcategory Label']
        .fillna('')
        .str.replace(r'^\d+\s*', '', regex=True)
        .str.strip()
        .replace({'': '--'})
    )
    
    df['brand_name'] = df['Brand Name'].apply(standardize_brand)
    df['item_name'] = df['Item Name']
    
    print(f"Loaded {len(df)} records from CSV")
    print(f"Brands in data: {df['brand_name'].unique()}")
    #---------------------------------------------------------------------------------
    
    print(f"Loaded {len(df)} records")
    
    # ===== TRAIN CRF MODEL =====
    print("Training CRF model...")
    crf_model = train_crf_size_extractor(df)
    print()
    
    # ===== TRAIN ML MODELS =====
    print("Training ML models...")
    ml_models, ml_vectorizers = train_ml_subbrand_classifier(df)
    print()
    
    # ===== SAVE ALL MODELS =====
    print("Saving models...")
    
    # Save CRF model
    crf_path = Path(__file__).parent / "crf_model.pkl"
    with open(crf_path, "wb") as f:
        pickle.dump(crf_model, f)
    
    # Save ML models
    ml_models_path = Path(__file__).parent / "models.pkl"
    with open(ml_models_path, "wb") as f:
        pickle.dump(ml_models, f)
    
    # Save ML vectorizers
    ml_vectorizers_path = Path(__file__).parent / "vectorizers.pkl"
    with open(ml_vectorizers_path, "wb") as f:
        pickle.dump(ml_vectorizers, f)
        
    print(f"CRF model saved to {crf_path}")
    print(f"ML models saved to {ml_models_path}")
    print(f"ML vectorizers saved to {ml_vectorizers_path}")
    print()

if __name__ == "__main__":
    main()