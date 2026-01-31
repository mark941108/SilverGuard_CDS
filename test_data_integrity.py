#!/usr/bin/env python3
"""
🔬 MedGemma Data Integrity Verification Script
==============================================
Validates the drug database for common data quality issues.
"""

import sys
from collections import Counter

print("="*60)
print("🔬 MedGemma Data Integrity Verification")
print("="*60)

# Test 1: Check for duplicate codes
print("\n[1/3] Checking for duplicate drug codes...")
try:
    from medgemma_data import DRUG_DATABASE
    
    all_codes = []
    drug_catalog = []
    
    for category, drugs in DRUG_DATABASE.items():
        for drug in drugs:
            all_codes.append(drug['code'])
            drug_catalog.append({
                'code': drug['code'],
                'name': drug['name_en'],
                'category': category
            })
    
    # Find duplicates
    code_counts = Counter(all_codes)
    duplicates = {code: count for code, count in code_counts.items() if count > 1}
    
    if duplicates:
        print(f"  ❌ FAILED: Found {len(duplicates)} duplicate code(s):")
        for dup_code, count in duplicates.items():
            print(f"     Code {dup_code} appears {count} times:")
            for item in drug_catalog:
                if item['code'] == dup_code:
                    print(f"       - {item['name']} ({item['category']})")
        sys.exit(1)
    else:
        print(f"  ✅ PASSED: All {len(all_codes)} drug codes are unique")
        
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 2: Check required fields
print("\n[2/3] Checking required fields...")
try:
    required_fields = ['code', 'name_en', 'name_zh', 'generic', 'dose', 
                       'appearance', 'indication', 'warning', 'default_usage']
    
    missing_fields = []
    
    for category, drugs in DRUG_DATABASE.items():
        for drug in drugs:
            for field in required_fields:
                if field not in drug:
                    missing_fields.append({
                        'drug': drug.get('name_en', 'UNKNOWN'),
                        'category': category,
                        'field': field
                    })
    
    if missing_fields:
        print(f"  ❌ FAILED: Found {len(missing_fields)} missing field(s):")
        for item in missing_fields[:5]:  # Show first 5
            print(f"     - {item['drug']} ({item['category']}) missing '{item['field']}'")
        sys.exit(1)
    else:
        print(f"  ✅ PASSED: All drugs have all {len(required_fields)} required fields")
        
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 3: Test get_renderable_data function
print("\n[3/3] Testing get_renderable_data() function...")
try:
    from medgemma_data import get_renderable_data
    
    data = get_renderable_data()
    
    # Check expected categories
    expected_categories = ['SOUND_ALIKE_CRITICAL', 'LOOK_ALIKE_SHAPE', 'GENERAL_TRAINING']
    missing_categories = [cat for cat in expected_categories if cat not in data]
    
    if missing_categories:
        print(f"  ❌ FAILED: Missing categories: {missing_categories}")
        sys.exit(1)
    
    # Check each category has drugs
    empty_categories = [cat for cat, drugs in data.items() if len(drugs) == 0]
    
    if empty_categories:
        print(f"  ❌ FAILED: Empty categories: {empty_categories}")
        sys.exit(1)
    
    total_drugs = sum(len(drugs) for drugs in data.values())
    print(f"  ✅ PASSED: get_renderable_data() returns {total_drugs} drugs across {len(data)} categories")
    
    for cat, drugs in data.items():
        print(f"     - {cat}: {len(drugs)} drugs")
    
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Final Report
print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Database Integrity Verified!")
print("="*60)
print(f"\n📊 Database Statistics:")
print(f"   Total Drugs: {len(all_codes)}")
print(f"   Categories: {len(DRUG_DATABASE)}")
print(f"   Average Drugs per Category: {len(all_codes) / len(DRUG_DATABASE):.1f}")
print("\n🚀 Database is ready for production use!")
sys.exit(0)
