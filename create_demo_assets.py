from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('examples', exist_ok=True)

def create_placeholder(filename, text, color="white"):
    img = Image.new('RGB', (896, 896), color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, otherwise use default
    try:
        # Try common linux font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    except:
        try:
             # Try windows font
             font = ImageFont.truetype("arial.ttf", 60)
        except:
             font = ImageFont.load_default()
    
    # Draw text roughly centered
    lines = text.split('\n')
    y = 300
    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        w = right - left
        h = bottom - top
        x = (896 - w) / 2
        draw.text((x, y), line, fill='black', font=font)
        y += h + 20
        
    img.save(f"examples/{filename}")
    print(f"Generated: examples/{filename}")

create_placeholder('safe_metformin.png', 'SAFE PRESCRIPTION\nMetformin 500mg\nTake with meals', '#e8f5e9')
create_placeholder('high_risk_elderly.png', 'HIGH RISK\nGlimepiride 4mg\nPatient Age: 85', '#ffebee')
create_placeholder('blurry_reject.png', 'BLURRY IMAGE\n(Too Blurry)', '#eeeeee')
