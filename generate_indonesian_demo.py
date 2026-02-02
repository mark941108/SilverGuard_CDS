from gtts import gTTS
import os

def generate_indonesian_demo():
    """Reserved for Red Team Audit P0-6: Indonesian TTS Generation"""
    print("🔊 Generating Indonesian Demo Assets...")
    
    demo_texts = {
        'warning': "MOHON TANYA APOTEKER. Dosis obat terlalu tinggi untuk usia pasien.",
        'safe': "Obat aman untuk diminum sesuai petunjuk dokter.",
        'instruction': "Minum obat setelah makan pagi, satu kali sehari."
    }
    
    os.makedirs("demo_assets", exist_ok=True)
    
    for key, text in demo_texts.items():
        print(f"   ... Synthesizing [{key}]: {text}")
        tts = gTTS(text=text, lang='id')
        filename = f"demo_assets/demo_indonesian_{key}.mp3"
        tts.save(filename)
        print(f"   ✅ Saved: {filename}")

if __name__ == "__main__":
    generate_indonesian_demo()
