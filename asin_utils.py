import re
from typing import Any
import pickle
from pathlib import Path

# Locate pickle files
BASE = Path(__file__).parent
ML_MODELS = {}
ML_VECS = {}

try:
    with open(BASE/"models.pkl", "rb") as f: 
        ML_MODELS = pickle.load(f)
    print("Loaded ML models")
except FileNotFoundError:
    print("models.pkl not found - will use fallback methods")
    ML_MODELS = {}

try:
    with open(BASE/"vectorizers.pkl","rb") as f: 
        ML_VECS = pickle.load(f)
    print("Loaded ML vectorizers") 
except FileNotFoundError:
    print("vectorizers.pkl not found - will use fallback methods")
    ML_VECS = {}
    # Add this after the ML model loading:
try:
    with open(BASE/"crf_model.pkl", "rb") as f: 
        CRF_MODEL = pickle.load(f)
    print("Loaded CRF model")
except FileNotFoundError:
    CRF_MODEL = None
    print("crf_model.pkl not found - will use regex-only size extraction")
    

def predict_sub_brand_ml(row):
    """Fast inference: look up the pre‐trained model & vectorizer."""
    brand = standardize_brand(row['brand_name'])
    if brand not in ML_MODELS:
        return map_subcategory_with_keywords(row)

    model = ML_MODELS[brand]
    
    # Check if the model is just a single class case
    if isinstance(model, str):
        return model 
    
    # vectorizer if we have an actual ML model
    if brand not in ML_VECS:
        # fallback to rule‐based if no vectorizer
        return map_subcategory_with_keywords(row)
        
    vec = ML_VECS[brand]

    # build the same text feature you used at train time:
    feat = clean_text(row['original_sub_brand']) + " " + clean_text(row['item_name'])
    return model.predict(vec.transform([feat]))[0]

# 1) normalize_text + get_silver_span
def normalize_text(name: str) -> str:
    text = name or ""
    text = re.sub(r"[-–]", " ", text)
    text = re.sub(r"\b(fluid)?\s*ounces?\b", "fl oz", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\d)\.(\d+)", r"0.\1", text)
    return text

def get_silver_span(name: str) -> str:
    """Extract SIZE information (volume, weight, dimensions) - NOT count/pack quantities"""
    text = normalize_text(name)
    
    text_cleaned = re.sub(r"\b\d+\s*-?\s*in\s*-?\s*\d+\b", "", text, flags=re.IGNORECASE)
    
    size_units = (
        r"fl\.?\s*oz|fluid\s*ounces?|ounces?|oz"
        r"|ml|millilit(?:e|er)s?"
        r"|l|litres?|liters?"
        r"|g|grams?|kg|mg"
        r"|inch(?:es)?|in"
        r"|yd(?:s)?|yard(?:s)?"
        r"|pound(?:s)?|lb(?:s)?"
    )
    
    # Priority 1: "(N UNIT Total)" 
    total_match = re.search(rf"\((\d*\.?\d+)\s*({size_units})\s*Total\)", text_cleaned, flags=re.IGNORECASE)
    if total_match:
        return total_match.group(0)
    
    # Priority 2: Find all potential sizes and pick the best one
    all_matches = []
    
    # Multiplicative "N x M UNIT" (for volumes/weights)
    for match in re.finditer(rf"(\d+)\s*[×xX]\s*(\d*\.?\d+)\s*({size_units})\b", text_cleaned, flags=re.IGNORECASE):
        all_matches.append((match.group(0), 3))
    
    # Dimension "N UNIT by M UNIT"
    for match in re.finditer(rf"(\d*\.?\d+)\s*({size_units})\s*(?:by|x|×)\s*(\d*\.?\d+)\s*({size_units})", text_cleaned, flags=re.IGNORECASE):
        all_matches.append((match.group(0), 3))
    
    # Simple "N UNIT" - but filter out small dosages and prioritize larger volumes
    for match in re.finditer(rf"(\d*\.?\d+)\s*({size_units})\b", text_cleaned, flags=re.IGNORECASE):
        matched_text = match.group(0)
        number = float(match.group(1))
        unit = match.group(2).lower()
            
        # Prioritize larger volumes/sizes
        if unit in ['fl oz', 'oz', 'ml', 'l', 'inch', 'in', 'yd']:
            all_matches.append((matched_text, 2))
        else:
            all_matches.append((matched_text, 1))
    
    # Return the highest priority match
    if all_matches:
        best_match = max(all_matches, key=lambda x: x[1])
        return best_match[0]
    
    return ""  # No size found

# 2) get_packsize
def get_packsize(name: str) -> str:
    """Extract PACK SIZE (quantity of items) - NOT physical dimensions"""
    text = normalize_text(name)
    
    # Pattern 1: "(Pack of N)" 
    pack_of_match = re.search(r"\(pack\s+of\s+(\d+)\)", text, flags=re.IGNORECASE)
    if pack_of_match:
        return pack_of_match.group(1)
    
    # Pattern 2: "N x M-strips" or "N x M ct" - extract N 
    mult_match = re.search(r"(\d+)\s*x\s*\d+\s*[-\s]?(?:strips?|ct|count|caplets?)", text, flags=re.IGNORECASE)
    if mult_match:
        return mult_match.group(1)
    
    # Pattern 3: "N Pack" or "Pack of N" patterns (but not simple counts)
    pack_patterns = [
        r"(\d+)\s*pack\b(?!\s*of)",  # "3 Pack" but not "Pack of 3"
        r"pack\s+of\s+(\d+)",       # "Pack of 3"
        r"(\d+)\s*-?\s*pack\b"      # "3-Pack"
    ]
    
    for pattern in pack_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Pattern 4: Special multi-pack patterns like "3 x 20 ct", "2 x 25 ct"
    multi_pack = re.search(r"(\d+)\s*x\s*\d+\s*ct\b", text, flags=re.IGNORECASE)
    if multi_pack:
        return multi_pack.group(1)
    
    # Pattern 5: Look for explicit "N Pks" or "N Items" 
    explicit_multi = re.search(r"(\d+)\s*(?:pks?|items?)\b", text, flags=re.IGNORECASE)
    if explicit_multi:
        count = int(explicit_multi.group(1))
        if count > 1:  # Only return if it's actually multiple items
            return str(count)
    
    # Pattern 6: Single item counts - but be more selective
    # Only extract count if it seems reasonable as a pack size AND there's no physical size
    count_match = re.search(r"(\d+)\s*(?:ct|count)\b", text, flags=re.IGNORECASE)
    if count_match:
        count_val = int(count_match.group(1))
        
        # Check if there's also a physical size measurement in the text
        has_physical_size = bool(re.search(r"\d+(?:\.\d+)?\s*(?:fl\s*oz|oz|ml|l|mg|g|kg|inch|in|yd|lb)\b", text, flags=re.IGNORECASE))
        
        # If there's a physical size AND this is a reasonable pack count, use it
        # If there's NO physical size, this count IS the pack size
        if not has_physical_size or (2 <= count_val <= 100):
            return str(count_val)
    
    # Pattern 7: Decimal counts like "1.0 Count" - convert to integer
    decimal_count = re.search(r"(\d+(?:\.\d+)?)\s+count", text, flags=re.IGNORECASE)
    if decimal_count:
        count_val = float(decimal_count.group(1))
        if count_val > 1:
            return str(int(count_val)) if count_val.is_integer() else str(count_val)
    
    return "1"  # Default to 1 if no pack size found

# 3) standardize_brand
def standardize_brand(brand):
    """Standardize brand names to ensure consistent mapping"""
    brand = brand.lower().strip()
    
    if brand in ["aveeno", "aveeno baby"]:
        return "aveeno"
    if brand in ["zarbee's", "zarbee's naturals", "zarbees naturals", "zarbees"]:
        return "zarbees"
    if brand in ["johnson's baby", "johnsons baby", "johnson and johnson", "j baby"]:
        return "j baby"
    if brand in ["maui moisture", "maui"]:
        return "maui"
    if brand in ["rogaine", "rogain"]:
        return "rogaine"
    if brand == "lubriderm":
        return "lubriderm"
    if brand == "ogx":
        return "ogx"
    if brand == "tylenol":
        return "tylenol"
    if brand == "bandaid":
        return "bandaid"
    if brand == "benadryl":
        return "benadryl"
    if brand == "imodium":
        return "imodium"
    if brand == "lactaid":
        return "lactaid"
    if brand == "listerine":
        return "listerine"
    if brand == "motrin":
        return "motrin"
    if brand == "neutrogena":
        return "neutrogena"
    if brand == "pepcid":
        return "pepcid"
    if brand == "sudafed":
        return "sudafed"
    if brand == "zyrtec":
        return "zyrtec"
    
    return brand

# 4) map_subcategory_with_keywords
def map_subcategory_with_keywords(row):
    """Map original sub-brand to standardized sub-brand using keywords"""
    brand = standardize_brand(row['brand_name']) 
    original_sub_brand = row['original_sub_brand']
    item_name = row['item_name'] if 'item_name' in row else ''
    
    # Special case for brand will be modeling in total level 
    if brand in {'bandaid', 'maui', 'imodium', 'lubriderm', 'rogaine'}:
        return brand.capitalize()
 
    # Get brand's standard sub-brands
    brand_mapping = get_brand_subbrand_mapping()
    if brand not in brand_mapping:
        return original_sub_brand  # Return original if brand not in our mapping
        
    # Special case for Aveeno Baby products
    if brand == "aveeno":
        item_text = (item_name or "").lower()
        # If "baby" appears in the item name or original sub-brand, assign to "Baby" sub-brand
        if "baby" in item_text or (original_sub_brand and "baby" in original_sub_brand.lower()):
            return "Baby"
    
    # Get keyword mapping for this brand
    keyword_mapping = get_keyword_mapping()
    
    # If we have keywords for this brand
    if brand in keyword_mapping:
        # Combine original sub-brand and item name for better matching
        text = f"{original_sub_brand} {item_name}".lower()
        
        # Check each sub-brand's keywords
        for sub_brand, keywords in keyword_mapping[brand].items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return sub_brand
    
    # If we have the brand but no keyword match, use fallback logic
    if brand == "Neutrogena":
        return "N/A"  # Default for Neutrogena
    elif brand == "Aveeno":
        return "N/A"  # Default for Aveeno
    elif brand == "Zarbees":
        return "N/A"  # Default for Zarbees
    elif brand == "Zyrtec":
        return "N/A"  # Default for Zyrtec
    elif brand == "Benadryl":
        return "N/A"  # Default for Benadryl
    elif brand == "Bandaid":
        return "Bandaid"  # Default for Bandaid
    elif brand == "Sudafed":
        return "N/A"  # Default for Sudafed
    elif brand == "OGX":
        return "All Other"  # Default for OGX
    elif brand == "Tylenol":
        return "Extra Strength"  # Default for Tylenol
    elif brand == "Listerine":
        return "All Other Mouthwash" #Default for listerine 
    elif brand == "J Baby":
        return "All Other"  # Default for J Baby
    elif brand == "ROGAINE":
        return "N/A"  # Default for ROGAINE
    
    # Return first standard sub-brand as default if available
    if brand in brand_mapping and brand_mapping[brand]:
        return brand_mapping[brand][0]
    
    # If all else fails, return the original
    return original_sub_brand

def get_brand_subbrand_mapping():
    """Define the valid sub-brands for each brand"""
    return {
        "neutrogena": ["Acne", "Cleansing", "Makeup", "Moisturizers", "Treatments", "Sun", "Hair", "Body", "Men"],
        "aveeno": ["Baby", "Kids", "Daily Moisturizing", "Skin Relief", "All Other Body", "Face", "Sun", "Hair"],
        "tylenol": ["Extra Strength", "Precise", "Easy to Swallow", "Rapid Release Gel", "Arthritis", "PM", 
                   "Upper Respiratory", "Peds", "Supplements"],
        "motrin": ["Adults", "Peds"],
        "zarbees": ["Sleep", "Immune", "Upper Respiratory", "Wellness", "Vitamins"],
        "zyrtec": ["Adults", "Peds"],
        "benadryl": ["Adults", "Peds", "Topical"],
        "lactaid": ["Non-Protein Milk (EXCLUDING Calcium)", "NEW - Calcium Milk", "Protein Milk", "Ice Cream", "Supplements"],
        "pepcid": ["AC", "Complete", "Gummies", "Max"],
        "listerine": ["Cool Mint", "Total Care", "Portables", "All Other Mouthwash"],
        "imodium": ["N/A"],
        "bandaid": ["N/A"],
        "sudafed": ["Adults", "Peds"],
        "ogx": ["Shampoo+Conditioner", "Stylers+Treatment", "All Other"],
        "j baby": ["Classic", "All Other"],
        "maui": ["N/A"],
        "rogaine": ["N/A"]
    }

def get_keyword_mapping():
    """Define keywords that map to each sub-brand by brand"""
    return {
        "neutrogena": {
            "Acne": [
                "acne", "pimple", "blemish", "breakout", "blackhead", "prone skin",
                "salicylic acid", "benzoyl peroxide", "rapid clear", "body clear",
                "stubborn acne", "stubborn marks", "oil-free acne"
            ],
            "Cleansing": [
                "cleanser", "cleansing", "wash", "scrub", "toner", "micellar", 
                "wipe", "towelette", "makeup remov", "removing", "remover",
                "purifying", "exfoliat", "deep clean", "peel off", "pore", "clay",
                "fresh foaming", "facial"
            ],
            "Makeup": [
                "concealer", "lip shine", "hydrating lip", "hydrating concealer", "lip"
            ],
            "Sun": [
                "spf", "sunscreen", "sun protection", "uv protection", "uv tint", 
                "broad spectrum", "zinc", "mineral uv", "purescreen", "sheer zinc",
                "after sun", "uva/uvb", "water resistant", "clear face", 
                "ultra sheer", "invisible daily defense", "sun rescue"
            ],
            "Hair": [
                "shampoo", "conditioner", "anti residue", "dandruff", "t/gel", 
                "scalp", "hair", "healthy scalp", "clarify & shine", "anti residue"
            ],
            "Moisturizers": [
                "moisturizer", "cream", "lotion", "gel cream", "water gel", "hydro boost", 
                "night cream", "face lotion", "hydrating", "hydrator", "gel-cream", 
                "norwegian formula", "oil free face moisturizer", "hydrating face", 
                "moisture", "deep moisture"
            ],
            "Treatments": [
                "serum", "retinol", "repair", "bright boost", "firm", "peptide", "anti aging",
                "wrinkle", "tone repair", "triple age", "eye cream", "vitamin c", "collagen",
                "niacinamide", "hyaluronic acid", "glycolic acid", "overnight", 
                "rapid wrinkle", "rapid tone", "rapid firming", "anti wrinkle"
            ],
            "Body": [
                "shower gel", "bath", "body oil", "body lotion", "body cream", "gel body", 
                "hand cream", "hand gel", "massage", "rainbath", "sesame", "body clear",
                "body wash", "body oil", "hydro boost body", "body mist", "feet", "foot", "hand", "nail"
            ],
            "Men": ["men", "unisex"]
        },
        "aveeno": {
            "Baby": ["baby", "infant", "newborn"],
            "Kids": ["kids", "children", "child"],
            "Daily Moisturizing": ["daily moistur", "daily lotion", "daily cream", "moisturizing lotion", "moisturizer"],
            "Skin Relief": [
                "skin relief", "relief", "eczema", "irritation", "sensitive skin",
                "therapeutic", "colloidal oatmeal", "shave gel", "foot mask", "hand mask",
                "repairing", "cica", "repair", "dry skin", "prebiotic"
            ],
            "All Other Body": ["body", "body wash", "body lotion", "shower"],
            "Face": ["face", "facial"],
            "Sun": ["sun", "spf", "sunscreen", "sunblock", "uv"],
            "Hair": ["hair", "shampoo", "conditioner"]
        },
        "ogx": {
            "Shampoo+Conditioner": ["shampoo", "conditioner"],
            "Stylers+Treatment": [
                "styler", "treatment", "mask", "serum", "oil", "cream",
                "styling", "curl", "curls", "mousse", "mist", 
                "spray", "thermal", "blowout", "heat protect", 
                "air dry", "finishing", "hold", "glue", "toning", 
                "drops", "repair", "protect", "bond", "protein"
            ],
            "All Other": ["other", "body", "face", "scrub"]
        },
        "j baby": {  
            "Classic": ["classic", "original", "traditional"],
            "All Other": ["other", "lotion", "shampoo", "wash", "oil", "powder"]
        },
        "ROGAINE": {
            "N/A": ["n/a", "hair", "minoxidil", "men", "women", "foam", "solution"]
        },
        "tylenol": {
            "Peds":             ["children", "child", "kid", "infant", "pediatric"],
            "PM":               ["pm", "nighttime", "sleep aid"],
            "Arthritis":        ["arthritis"],
            "Extra Strength":   ["extra strength"],
            "Rapid Release Gel":["rapid release", "gel"],
            "Easy to Swallow":  ["easy to swallow"],
            "Precise":          ["precise"],
            "Upper Respiratory":["sinus", "cold", "flu", "decongest", "mucus", "cough"],
            "Supplements":      ["supplement", "vitamin", "mineral"]
        },
        "listerine": {
            "Cool Mint": [
                "cool mint", "freshburst", "arctic mint", "glacier mint", "mint shield", "polar mint",
                "spearmint", "cinnamon mint", "mint flavor"
            ],
            "Total Care": [
                "total care", "smart rinse", "gum therapy", "sensitivity", "healthy white",
                "essential care", "clinical solutions", "nightly reset"
            ],
            "Portables": [
                "pocketmist", "pocketpaks", "tabs", "ready! tabs", "chewable", "mist", "travel size"
            ],
            "All Other Mouthwash": [
                "mouthwash", "antiseptic", "zero alcohol", "alcohol-free", "fluoride"
            ]
        },
        "lactaid": {
        "NEW - Calcium Milk": ["calcium"],
        "Protein Milk": ["protein"],
        "Non-Protein Milk (EXCLUDING Calcium)": [
            "milk", "fat free", "reduced fat", "low fat", "whole", "eggnog",
            "sour cream", "cottage cheese"
        ],
        "Ice Cream": ["ice cream"],
        "Supplements": ["caplet", "chewable", "enzyme", "lactase", "supplement"]
    },
        "benadryl": {
            "Peds": ["children", "kids", "child", "pediatric", "kidz", "bubble gum", "cherry flavor", "grape"],
            "Topical": [
                "topical", "cream", "gel", "spray", "itch relief", "anti-itch", "stick", 
                "extra strength", "cooling", "outdoor itches", "poison ivy", "insect bites", 
                "bug bite", "itch stopping", "skin itching", "skin protectant", "sunburn", 
                "analgesic", "zinc acetate"
            ],
            "Adults": [
                "ultratabs", "liqui-gels", "allergy relief", "antihistamine", "diphenhydramine hcl",
                "cold relief", "symptom relief", "cold & allergy", "dye free", "24 hour", 
                "allergy medicine", "allergy syrup", "sneezing", "runny nose", "itchy eyes", 
                "go packs", "travel", "on-the-go", "relief tablets", "allergy plus congestion",
                "nasal decongestant", "phenylephrine"
            ]
        },
        "motrin": {
            "Peds": [
                "children", "child", "kids", "infant", "pediatric", "infants", "childrens", "babies",
                "bubble gum", "berry flavor", "grape flavor", "concentrated drops", "dye-free", 
                "100mg ibuprofen", "oral suspension", "chewable tablets"
            ],
            "Adults": [
                "ib", "pm", "200 mg", "caplets", "tablets", "liquid gels", "migraine", 
                "pain reliever", "fever reducer", "muscle aches", "headache", "back pain",
                "menstrual cramps", "arthritis pain", "sleep aid", "dual action", "tylenol",
                "nsaid", "ibuprofen 200mg", "backache", "unit dose", "diclofenac sodium",
                "topical gel", "anti-inflammatory", "hands, wrists, elbows, knees"
            ]
        },
        "sudafed": {
            "Peds": [
                "children", "child", "kids", "pediatric", "children's", "berry-flavored",
                "alcohol free", "sugar-free", "for all ages", "berry", "liquid", 
                "children's pe", "kid", "infant"
            ],
            "Adults": [
                "pe", "sinus congestion", "maximum strength", "non-drowsy", "tablets", 
                "head congestion", "mucus relief", "sinus pressure", "caplets",
                "day and night", "pain relief", "nasal spray", "decongestant",
                "phenylephrine hcl", "ibuprofen", "acetaminophen", "guaifenesin",
                "flu severe", "antihistamine", "diphenhydramine", "oxymetazoline",
                "nasal mist", "nose spray", "congestion relief", "sinus severe", 
                "tablets for adults"
            ]
        },
        "zyrtec": {
            "Peds": [
                "children", "kids", "children's", "children's zyrtec", "bubble gum", 
                "grape flavor", "2.5 mg", "5 mg", "children's allergy", "dissolving",
                "chewables", "syrup", "dye-free", "sugar-free", "citrus flavor",
                "for kids", "2 years & older", "6 years & older"
            ],
            "Adults": [
                "10 mg", "24 hour", "tablets", "allergy relief", "cetirizine hcl", 
                "antihistamine", "indoor & outdoor", "liquid gels", "capsules",
                "runny nose", "sneezing", "itchy eyes", "allergy medicine",
                "ragweed", "tree pollen", "relief", "travel size", "value pack",
                "indoor and outdoor", "bonus pack", "dispensit pack"
            ]
        },
        "pepcid": {
                    "AC": [
                        "pepcid ac original", "pepcid ac maximum", "pepcid ac max"],
                    "Complete": ["pepcid complete", "acid reducer + antacid", "dual action"],
                    "Gummies": ["pepcid soothing gummies", "gummies", "stress relief"],
                    "Max": ["icy cool mint", "icy mint flavor"]
        },
        "zarbees": {
            "Sleep": [
                "melatonin", "sleep", "bedtime", "nighttime", "sleep supplement", 
                "sleep aid", "gentle bedtime", "bedtime spray", "sleep spray",
                "calming", "chamomile", "lavender", "l-theanine", "gaba",
                "peaceful sleep", "unwind", "relax", "prepare for sleep",
                "calm mind", "calm body", "massage oil", "massage balm"
            ],
            "Immune": [
                "immune", "immunity", "immune support", "elderberry", "vitamin c",
                "vitamin d", "zinc", "daily immune", "total immune", "black elderberry",
                "elderberry syrup", "elderberry gummies", "immune gummies",
                "immune vitamins", "daily immunity", "immune support syrup"
            ],
            "Upper Respiratory": [
                "cough", "cough syrup", "mucus", "throat", "chest rub", "sinus",
                "respiratory", "nasal spray", "nasal saline", "soothing chest rub",
                "cough soothers", "throat relief", "sinus & respiratory", "ivy leaf",
                "thyme", "honey cough", "dark honey", "eucalyptus", "all-in-one"
            ],
            "Wellness": [
                "probiotic", "digestive", "digestive health", "digestive support",
                "multivitamin", "complete multivitamin", "daily multivitamin",
                "regularity support", "prebiotic fiber", "constipation", "gum massage",
                "teething", "bottom balm", "face balm", "vitamin drops", "calm gummy"
            ],
            "Vitamins": [
                "vitamins", "vitamin d drops", "toddler vitamins", "complete vitamins",
                "multivitamin", "b-complex", "vitamin a", "vitamin b", "vitamin e",
                "folic acid", "daily vitamins", "supplement", "baby vitamins",
                "children vitamins", "kids vitamins"
            ]
        }       
}

def clean_text(text):
    """Clean and normalize text for better matching"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    """Split into words (with dot) or any single char (so hyphens separate)."""
    return re.findall(r"\w+\.?\w*|\S", text)

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

def predict_size_with_crf(name: str) -> str:
    """Use CRF model for size extraction, fall back to regex"""
    if CRF_MODEL is None:
        return get_silver_span(name)
    
    # Use CRF prediction logic here
    norm = normalize_text(name)
    toks = tokenize(norm)
    features = [token2features(toks, i) for i in range(len(toks))]
    tags = CRF_MODEL.predict([features])[0]
    
    # Extract SIZE tokens
    size_toks = [tok for tok, tag in zip(toks, tags) if tag.endswith("SIZE")]
    pred = " ".join(size_toks).strip()
    
    # Fall back to regex if CRF finds nothing
    if not pred:
        pred = get_silver_span(name)
    
    return pred