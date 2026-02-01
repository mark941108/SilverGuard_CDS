# Test: medication calendar generation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from HF_SPACE_APP import create_medication_calendar

# Test Case 1: Normal drug
test_case_1 = {
    "extracted_data": {
        "drug": {"name": "Aspirin 100mg", "dose": "100mg"},
        "usage_instructions": {"quantity": "28", "timing": "早晨", "route": "口服"}
    },
    "safety_analysis": {"status": "PASS", "detected_issues": []}
}

# Test Case 2: High Risk
test_case_2 = {
    "extracted_data": {
        "drug": {"name": "Warfarin 5mg", "dose": "5mg"},
        "usage_instructions": {"quantity": "30", "timing": "睡前", "route": "口服"}
    },
    "safety_analysis": {
        "status": "HIGH_RISK",
        "detected_issues": ["出血風險增加"]
    }
}

print("\\nSilverGuard Calendar Test\\n" + "=" * 60)

# Run tests
for i, test_data in enumerate([test_case_1, test_case_2], 1):
    print("\\nTest", i)
    try:
        path = create_medication_calendar(test_data)
        img = Image.open(path)
        print("  PASS - Size:", img.size, "Mode:", img.mode)
        print("  File:", path)
    except Exception as e:
        print("  FAIL:", str(e))

print("\\nTest Complete!")
