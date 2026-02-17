# tts_engine.py
# 🏥 SilverGuard CDS: Multilingual TTS Engine (Google Cloud Hybrid)
import os
import sys
import time
import platform
import asyncio

# [Round 114] 雲端 TTS 支援
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# [Round 114] OS Detection
SYSTEM_OS = platform.system()

def generate_edge_tts(text, filename, lang):
    """
    ☁️ Cloud-Native TTS using Microsoft Edge Engine (No SAPI5 needed)
    """
    if not EDGE_TTS_AVAILABLE:
        print("⚠️ Edge-TTS not installed. Skipping.")
        return False

    # Map languages to Neural Voices
    VOICE_MAP = {
        "zh": "zh-TW-HsiaoChenNeural",
        "zh-TW": "zh-TW-HsiaoChenNeural",
        "en": "en-US-AriaNeural",
        "id": "id-ID-GadisNeural",
        "vi": "vi-VN-HoaiMyNeural"
    }
    
    # Resolve exact voice
    target_voice = VOICE_MAP.get(lang, "en-US-AriaNeural")
    if "zh" in lang: target_voice = VOICE_MAP["zh-TW"]
    elif "id" in lang: target_voice = VOICE_MAP.get("id", "id-ID-GadisNeural")
    elif "vi" in lang: target_voice = VOICE_MAP.get("vi", "vi-VN-HoaiMyNeural")

    async def _run_edge():
        communicate = edge_tts.Communicate(text, target_voice)
        await communicate.save(filename)

    try:
        # Handle Event Loops for Jupyter/Kaggle
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(_run_edge())
            else:
                asyncio.run(_run_edge())
        except Exception as loop_err:
            # Fallback for complex threading scenarios
             asyncio.run(_run_edge())
            
        return True
    except Exception as e:
        print(f"⚠️ Edge-TTS Error: {e}")
        return False

def tts_entry_point(text, filename, lang="zh-TW"):
    """
    🏭 Universal TTS Factory (Windows SAPI5 + Linux Edge-TTS)
    """
    requested_lang = str(lang).lower().split('-')[0]
    
    print(f"🚀 [TTS Worker] OS: {SYSTEM_OS} | Lang: {lang}")

    # --- Strategy A: Piper (High Quality Offline) - Cross Platform ---
    # Used for ID/VI if Piper is available locally
    # [Fix] Restrict to Windows only to prevent Linux (Kaggle) Exec format error
    if requested_lang in ["id", "vi"] and SYSTEM_OS == "Windows":
        try:
            import piper_engine
            # Check if piper_engine is actually set up (not just imported)
            # Assuming piper_engine exposes a check or we just try
            print(f"🚀 [TTS Worker] Routing to Piper Engine for '{requested_lang}'...")
            success = piper_engine.generate_speech_offline(text, filename, lang=requested_lang)
            if success:
                print(f"✅ [TTS Worker] Piper Success for '{requested_lang}'.")
                return 
            print(f"⚠️ [TTS Worker] Piper failed for '{requested_lang}', attempting OS fallback.")
        except ImportError:
            pass # Piper not installed
        except Exception as e:
            print(f"⚠️ [TTS Worker] Piper Integration Error: {e}")

    # --- Strategy B: OS-Specific Fallback ---
    if SYSTEM_OS == "Linux":
        # 🐧 Linux: Use Edge-TTS (Internet Required in Cloud)
        print("☁️ [Linux] Routing to Edge-TTS...")
        if generate_edge_tts(text, filename, lang):
            print(f"✅ [TTS] Edge-TTS success: {filename}")
            return
        else:
            print("❌ [TTS] All Linux engines failed.")

    elif SYSTEM_OS == "Windows":
        # 🪟 Windows: Use SAPI5 (Offline)
        try:
            # 🚨 CRITICAL: Only import pythoncom/pyttsx3 inside Windows block
            import pythoncom
            import pyttsx3
            
            pythoncom.CoInitialize()
            
            # Platform-agnostic initialization attempt
            try:
                engine = pyttsx3.init()
            except:
                engine = pyttsx3.init('sapi5')
                
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Voice Logic
            voices = engine.getProperty('voices')
            target_voice_id = None
            
            # Simple keyword matching for Windows voices
            keywords = {
                "zh": ["Chinese", "Han", "Taiwan", "Yating", "Hanhan"],
                "en": ["English", "David", "Zira", "US"],
                "id": ["Indonesia", "Andika", "Bahasa", "Gadis", "Wayan"],
                "vi": ["Vietnam", "An", "Tieng", "Hoai", "Nam"]
            }
            
            search_terms = keywords.get(requested_lang, keywords["en"])
            
            for v in voices:
                v_name = str(v.name).lower()
                if any(k.lower() in v_name for k in search_terms):
                    target_voice_id = v.id
                    break
            
            if target_voice_id:
                engine.setProperty('voice', target_voice_id)
            
            # Generate
            engine.save_to_file(text, filename)
            engine.runAndWait()
            print(f"✅ [TTS] SAPI5 success: {filename}")
            
        except Exception as e:
            print(f"❌ [Windows TTS Error] {e}")

if __name__ == "__main__":
    # Test
    tts_entry_point("Testing Cloud TTS", "test.mp3", "en")
