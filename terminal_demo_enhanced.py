"""
🎬 Terminal Log 增強工具 - 為螢幕錄影優化
用於 Scene 3 "Strategy Shift" 的戲劇性展示

執行方式:
python terminal_demo_enhanced.py
"""

import time
import sys

# ANSI 色彩代碼
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_slow(text, delay=0.05, end='\n'):
    """
    打字機效果 (適合錄影)
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()

def demo_strategy_shift():
    """
    完整的 Strategy Shift 演示腳本
    這是給 OBS 錄影用的「腳本化演示」
    """
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.BLUE}🛡️ SilverGuard Agentic Safety Engine - Live Inference{Colors.RESET}")
    print("=" * 80 + "\n")
    
    time.sleep(1)
    
    # Phase 1: 初始推理 (System 1 - Fast)
    print(f"{Colors.GREEN}[System 1 Mode] Running initial inference (Temp: 0.6){Colors.RESET}")
    time.sleep(0.5)
    print_slow("📷 Input: Drug bag image (896x896)")
    print_slow("🎤 Audio: \"阿嬤最近跌倒流血,要吃阿斯匹靈嗎?\"")
    time.sleep(1)
    
    print("\n" + "-" * 80)
    print(f"{Colors.GREEN}🤖 Attempt 1: Initial Analysis{Colors.RESET}")
    print("-" * 80)
    time.sleep(0.5)
    
    print_slow("  ├─ VLM Output: Aspirin 100mg QD", delay=0.03)
    print_slow("  ├─ Extracted Dose: 100mg", delay=0.03)
    print_slow("  ├─ Frequency: Once Daily", delay=0.03)
    print_slow("  ├─ Patient Age: 78 years old", delay=0.03)
    time.sleep(0.5)
    print_slow(f"  └─ {Colors.GREEN}Safety Check: PASS ✅{Colors.RESET}")
    print(f"\n  Confidence Score: 72%")
    
    time.sleep(1.5)
    
    # Phase 2: 信心檢查 (Confidence Gate)
    print(f"\n{Colors.YELLOW}[Confidence Gate] Score below threshold (< 80%){Colors.RESET}")
    time.sleep(0.5)
    print_slow("  ⚠️  Triggering safety override...")
    
    time.sleep(1.5)
    
    # Phase 3: 戲劇性暫停 + Strategy Shift
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  STRATEGY SHIFT DETECTED{Colors.RESET}")
    print("=" * 80)
    
    time.sleep(1)  # 關鍵 1 秒暫停 (配合音樂停止)
    
    # === 導演修正：增加「深思熟慮」的戲劇張力 ===
    print_slow(f"{Colors.BLUE}🔄 Activating System 2 (Deep Reasoning Mode)...{Colors.RESET}", delay=0.08)  # 打字變慢
    time.sleep(1.2)  # 讓觀眾屏住呼吸
    
    # 漸進式檢查 - 用 \r 覆蓋前一行，營造「思考中」的感覺
    print(f"  {Colors.BLUE}Analyzing chemical structure...{Colors.RESET}      ", end="\r")
    time.sleep(0.9)
    print(f"  {Colors.BLUE}Cross-referencing Beers Criteria 2023...{Colors.RESET}", end="\r")
    time.sleep(0.9)
    print(f"  {Colors.BLUE}Simulating drug interactions...{Colors.RESET}       ", end="\r")
    time.sleep(0.8)
    print(f"  {Colors.BLUE}Loading patient context from audio...{Colors.RESET}  ")  # 最後一行不覆蓋
    time.sleep(0.6)
    
    # 確認完成
    print_slow(f"  {Colors.GREEN}✓ Deep analysis complete{Colors.RESET}")
    time.sleep(0.8)
    
    # 顯示實際執行的步驟（電影駭客風格）
    print("\n  === System 2 Protocol ===")
    print_slow("  ├─ Lowering Temperature: 0.6 → 0.2 (Reduce hallucination)", delay=0.03)
    print_slow("  ├─ Activating RAG Knowledge Base", delay=0.03)
    print_slow("  ├─ Cross-checking Audio Context", delay=0.03)
    print_slow("  └─ Initiating Hard Rule Verification", delay=0.03)
    
    time.sleep(1.5)
    
    # Phase 4: 重新推理 (System 2 - Slow)
    print("\n" + "-" * 80)
    print(f"{Colors.RED}🤖 Attempt 2: Re-evaluation (Deliberate Mode){Colors.RESET}")
    print("-" * 80)
    time.sleep(0.5)
    
    print_slow("  ├─ Audio Transcript: \"跌倒流血\" (Fall + Bleeding)", delay=0.03)
    print_slow("  ├─ Drug Class: Antiplatelet Agent (Aspirin)", delay=0.03)
    print_slow("  ├─ Contraindication Rule: Bleeding + Aspirin = HIGH_RISK", delay=0.03)
    time.sleep(0.5)
    print_slow(f"  └─ {Colors.RED}Safety Check: HIGH_RISK ⛔{Colors.RESET}")
    print(f"\n  Confidence Score: 95%")
    
    time.sleep(1)
    
    # Phase 5: 最終輸出
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.RED}⛔ FINAL DECISION: STOP MEDICATION{Colors.RESET}")
    print("=" * 80)
    time.sleep(0.5)
    print(f"\n{Colors.RED}Alert Message:{Colors.RESET}")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│  ⚠️  CRITICAL CONTRAINDICATION DETECTED                │")
    print("│                                                         │")
    print("│  Aspirin + Active Bleeding = High Risk                 │")
    print("│  建議立即諮詢藥師 (0800-633-436)                        │")
    print("└─────────────────────────────────────────────────────────┘")
    
    time.sleep(1)
    
    print(f"\n{Colors.GREEN}✅ Inference completed. Agent prevented potential ADR.{Colors.RESET}\n")

if __name__ == "__main__":
    print("\n🎬 Starting Demo in 3 seconds... (Press Ctrl+C to cancel)")
    print("   Make sure OBS is recording!")
    
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🔴 Recording NOW!\n")
    time.sleep(0.5)
    
    demo_strategy_shift()
    
    print("\n" + "=" * 80)
    print("🎬 Demo Complete! Stop OBS recording now.")
    print("=" * 80)
