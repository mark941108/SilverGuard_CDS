
import re
import difflib

# COPY OF THE FIXED FUNCTION
def normalize_dose_to_mg(dose_str):
    """
    🧪 Helper: Normalize raw dosage string to milligrams (mg)
    Handles: "500 mg", "0.5 g", "1000 mcg"
    [V19 Update] Handles Ranges ("1-2 tabs") and Compounds ("160/12.5mg")
    Returns: (list_of_mg_values, is_valid_conversion)
    """
    if not dose_str: return [], False
    
    # Clean input
    s_full = str(dose_str).lower().replace(",", "").replace(" ", "")
    
    # [Audit Fix] Compound Dose Support: Split by / or +
    parts = re.split(r'[/\+]', s_full)
    results = []
    
    for s in parts:
        if not s: continue
        try:
            # Regex to find number + unit
            # [Audit Fix] Supports Chinese Units (毫克/公克)
            match = re.search(r'([\d\.]+)(mg|g|mcg|ug|ml|毫克|公克)', s)
            
            val = 0.0
            if not match:
                 # Fallback: strictly require unit or pure number if it looks like a dose
                 # [Audit Fix] Capture decimals in fallback
                 nums = re.findall(r'\d*\.?\d+', s)
                 if nums: 
                     val = float(nums[0]) # Raw number, assume mg if ambiguous but capture it
                 else:
                     continue # Skip unparseable parts
            else:
                val = float(match.group(1))
                unit = match.group(2)
                
                if unit in ['g', '公克']:
                    val *= 1000.0
                elif unit in ['mcg', 'ug']:
                    val /= 1000.0
                # else mg, ml, 毫克 -> keep as is
            
            results.append(val)
        except:
            continue
            
    if not results:
        return [], False
        
    return results, True

# Fuzzy Match Logic Simulation
def fuzzy_match(target, candidates):
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=0.8)
    return matches[0] if matches else None

def run_tests():
    print("🧪 Running Critical Logic Verification...")
    
    # 1. Compound Dose Tests
    cases = [
        ("160/12.5mg", [160.0, 12.5]),
        ("500/50 mg", [500.0, 50.0]),
        ("500mg", [500.0]),
        ("0.5g", [500.0]),
        ("1000mcg", [1.0]),
        ("invalid", []),
        ("500+2.5", [500.0, 2.5]), # Check + separator
    ]
    
    all_pass = True
    print("\n--- Dose Normalization Tests ---")
    for inp, expected in cases:
        res, valid = normalize_dose_to_mg(inp)
        # Check tolerance for float check
        match = False
        if len(res) == len(expected):
            match = all(abs(a-b) < 0.01 for a, b in zip(res, expected))
            
        status = "✅ PASS" if match else f"❌ FAIL (Got {res}, Exp {expected})"
        print(f"Input: '{inp}' -> {status}")
        if not match: all_pass = False

    # 2. Fuzzy Match Tests
    print("\n--- Fuzzy Synonym Tests ---")
    db = ["metformin", "glimepiride", "aspirin", "atorvastatin"]
    fuzzy_cases = [
        ("metformim", "metformin"), # Typo
        ("glimepirid", "glimepiride"), # Missing char
        ("asprin", "aspirin"), # Common typo
        ("tylenol", None) # No match
    ]
    
    for inp, expected in fuzzy_cases:
        res = fuzzy_match(inp, db)
        status = "✅ PASS" if res == expected else f"❌ FAIL (Got {res}, Exp {expected})"
        print(f"Input: '{inp}' -> {status}")
        if res != expected: all_pass = False

    if all_pass:
        print("\n✨ ALL TESTS PASSED. Logic is robust.")
    else:
        print("\n⚠️ SOME TESTS FAILED.")

if __name__ == "__main__":
    run_tests()
