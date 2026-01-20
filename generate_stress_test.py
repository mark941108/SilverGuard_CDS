"""
Gallery of Horrors - Stress Test Image Generator
=================================================
Generates 10 "nightmare" prescription images to demonstrate MedGemma's robustness.

Usage (run locally or on Kaggle):
    python generate_stress_test.py
    
Output: assets/stress_test/*.png (10 images)
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random
import numpy as np

# Create output directory
OUTPUT_DIR = "assets/stress_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample prescription data
SAMPLE_DRUGS = [
    {"name": "Metformin 500mg", "usage": "Take 1 tablet twice daily with meals", "chinese": "每日兩次，隨餐服用"},
    {"name": "Aspirin 100mg", "usage": "Take 1 tablet daily", "chinese": "每日一次，飯後服用"},
    {"name": "Lisinopril 10mg", "usage": "Take 1 tablet in the morning", "chinese": "每日早晨服用一錠"},
    {"name": "Warfarin 5mg", "usage": "Take as directed by doctor", "chinese": "依醫師指示服用"},
    {"name": "Atorvastatin 20mg", "usage": "Take 1 tablet at bedtime", "chinese": "睡前服用一錠"},
]

def create_base_prescription(drug_info, width=600, height=400):
    """Create a clean prescription label image"""
    # Cream/white background with slight texture
    img = Image.new('RGB', (width, height), color=(255, 252, 245))
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to PIL default
    try:
        font_large = ImageFont.truetype("arial.ttf", 28)
        font_med = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw prescription content
    y_offset = 30
    
    # Hospital header
    draw.text((20, y_offset), "🏥 Taiwan General Hospital", fill=(0, 80, 160), font=font_large)
    y_offset += 50
    
    # Patient info
    draw.text((20, y_offset), f"Patient: WU CHEN-MING (吳振明)", fill=(0, 0, 0), font=font_med)
    y_offset += 35
    draw.text((20, y_offset), f"Age: 78 | DOB: 1948-05-12", fill=(80, 80, 80), font=font_small)
    y_offset += 40
    
    # Drug name (large, prominent)
    draw.rectangle([(15, y_offset-5), (width-15, y_offset+45)], outline=(0, 100, 200), width=2)
    draw.text((25, y_offset+5), f"💊 {drug_info['name']}", fill=(0, 0, 0), font=font_large)
    y_offset += 60
    
    # Usage instructions
    draw.text((20, y_offset), f"Usage: {drug_info['usage']}", fill=(0, 0, 0), font=font_med)
    y_offset += 35
    draw.text((20, y_offset), f"用法: {drug_info['chinese']}", fill=(60, 60, 60), font=font_med)
    y_offset += 45
    
    # Warning box
    draw.rectangle([(15, y_offset), (width-15, y_offset+50)], fill=(255, 245, 230), outline=(255, 150, 0), width=2)
    draw.text((25, y_offset+15), "⚠️ Keep out of reach of children", fill=(200, 100, 0), font=font_small)
    
    return img


def apply_extreme_blur(img, intensity="heavy"):
    """Apply motion or gaussian blur"""
    if intensity == "heavy":
        return img.filter(ImageFilter.GaussianBlur(radius=8))
    elif intensity == "motion":
        # Simulate motion blur by box blur + directional offset
        return img.filter(ImageFilter.BoxBlur(radius=6))
    return img.filter(ImageFilter.GaussianBlur(radius=4))


def apply_low_light(img):
    """Simulate dark/underexposed image"""
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.35)  # Very dark
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(0.7)


def apply_overexposure(img):
    """Simulate washed out/overexposed image"""
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.8)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(0.5)


def apply_noise(img, amount=50):
    """Add random noise/grain"""
    arr = np.array(img)
    noise = np.random.randint(-amount, amount, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_crease(img):
    """Simulate paper fold/crease with a dark line"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    # Diagonal crease line
    draw.line([(0, height//3), (width, height//2)], fill=(180, 170, 160), width=8)
    # Make area around crease slightly darker
    return img


def apply_water_damage(img):
    """Simulate water stain/damage"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    # Draw irregular water stain shapes
    for _ in range(3):
        x = random.randint(0, width)
        y = random.randint(0, height//2)
        r = random.randint(40, 100)
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(220, 210, 190, 128))
    return img


def apply_skew(img, angle=15):
    """Rotate image to simulate camera angle"""
    return img.rotate(angle, expand=True, fillcolor=(240, 240, 240))


def apply_partial_occlusion(img):
    """Simulate finger partially covering text"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    # Draw a finger-shaped ellipse
    draw.ellipse([(width-150, height//2-60), (width+50, height//2+80)], fill=(210, 180, 160))
    return img


def apply_faded_label(img):
    """Simulate old, faded print"""
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.4)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(0.3)


# Generate 10 nightmare images
STRESS_TESTS = [
    ("01_extreme_blur", "Extreme Gaussian Blur", lambda img: apply_extreme_blur(img, "heavy")),
    ("02_motion_blur", "Motion Blur (shaky hands)", lambda img: apply_extreme_blur(img, "motion")),
    ("03_low_light", "Dark/Low Light Condition", apply_low_light),
    ("04_overexposed", "Overexposed/Washed Out", apply_overexposure),
    ("05_heavy_noise", "Heavy Noise/Grain", lambda img: apply_noise(img, 80)),
    ("06_paper_crease", "Paper Fold/Crease", apply_crease),
    ("07_water_damage", "Water Damage Simulation", apply_water_damage),
    ("08_skewed_angle", "45° Camera Angle", lambda img: apply_skew(img, 25)),
    ("09_finger_occlusion", "Partial Finger Occlusion", apply_partial_occlusion),
    ("10_faded_old", "Faded/Old Label", apply_faded_label),
]


if __name__ == "__main__":
    print("🎭 Generating Gallery of Horrors (10 Stress Test Images)...")
    print("=" * 60)
    
    for i, (filename, description, transform_fn) in enumerate(STRESS_TESTS):
        # Pick a random drug for variety
        drug = SAMPLE_DRUGS[i % len(SAMPLE_DRUGS)]
        
        # Create clean base image
        base_img = create_base_prescription(drug)
        
        # Apply distortion
        stressed_img = transform_fn(base_img)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
        stressed_img.save(output_path)
        print(f"  ✅ {filename}.png - {description}")
    
    print("=" * 60)
    print(f"🎉 Done! Images saved to: {OUTPUT_DIR}/")
    print("\n📋 Next step: Add these to README.md as 'Robustness Gallery'")
