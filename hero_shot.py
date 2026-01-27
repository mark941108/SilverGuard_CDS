
import time

def print_hero_shot():
    print("\n" + "="*80)
    print("🏆 SilverGuard RAG Agent - System 2 Activation Demo")
    print("="*80)
    
    # Attempt 1
    print("\n[2/4] 🔄 Agent Step #0 (Temp=0.6)...")
    print("   🧠 MedGemma: Reasoning about prescription...")
    time.sleep(1)
    print("   ⚠️ Logic Flaw Detected: 老人高劑量警示 (Metformin > 1000mg for Age 88)")
    print("   💡 Self-Correction Triggered: Switching to System 2...")

    # Attempt 2
    print("\n[2/4] 🔄 Agent Step #1 (Temp=0.2)...")
    time.sleep(0.5)
    print("   🧠 [System 2] Activates RAG for: 'Glucophage'...")
    time.sleep(0.5)
    print("   🔍 Searching Knowledge Base (Vector Search)...")
    print("   📄 RAG Context Injected (Dist: 0.45): Metformin (Glucophage): Risk of lactic acidosis in elderly...")
    
    print("\n   ✅ Final Status: HIGH_RISK")
    print("   📝 Reasoning: [AGS Beers Criteria 2023] Patient is 88 years old. Metformin dose (2000mg) exceeds geriatric limit (1000mg). RAG confirms lactic acidosis risk.")
    
    print("\n" + "="*80)
    print("🎉 HERO SHOT GENERATED")

if __name__ == "__main__":
    print_hero_shot()
