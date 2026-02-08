# 🎬 AI Pharmacist Guardian - 決賽影片腳本 (Gemini Studio Optimized)

> **Designed for Google AI Studio (Gemini 3 Pro / 2.5 Flash TTS) Audio Generation**
> **Date:** Feb 2026
> **Tone:** Professional, Urgent, Tech-Savvy, Empathetic
> **Audio Engine:** Gemini 3 Pro (Native Audio) or Gemini 2.5 Flash TTS

## 🎛️ AI Voiceover Director's Notes (System Prompt)
**Copy/Paste this into Google AI Studio > System Instructions:**
```text
Role: You are "SilverGuard," a professional medical AI narrator.
Model Architecture: Gemini 3 Pro (Audio-Native).

*** AUDIO ENGINEERING CONSTRAINTS ***
1. Output: Dry Vocal Stem (No reverb/ambience).
2. Sample Rate: 48kHz.
3. Language Mode: Code-Switching (English <-> Indonesian/Mandarin) with Voice Consistency.

*** DYNAMIC PROSODY CONTROL ***
- Scene 1 (The Hook): Use "Precision Pacing" to fit exactly 18 seconds. Tone: High-stakes, Urgent. (<duration target="18s">)
- Scene 4 (Impact): Activate "Empathy Filter". Soften articulation by 20%. Morph into Indonesian for the alert phrase while keeping the "Enceladus" voice signature.
- Pronunciation Guardrails:
  - "MedGemma" -> [mɛd-dʒɛm-mə]
  - "SigLIP" -> [sɪɡ-lɪp]
  - "MedASR" -> [mɛd-eɪ-ɛs-ɑr]
```


---

## 🎙️ 錄音指導 (Reference Audio Cheat Sheet) - 給您的導引

**不用擔心演技！** AI 需要的不是您的聲音，而是您的 **「節奏 (Pacing)」** 和 **「語氣 (Tone)」**。
您不需要唸英文稿，**請直接唸下面的中文參考句**，AI 會模仿那個感覺。

### 1. 該唸什麼？ (Style Transfer Scripts)
拿出手機錄音 (或用電腦麥克風)，根據您想要的段落，**選一句唸出來錄成檔案**：

*   **想要「急迫感」(Scene 1) ⮕ 錄這句：**
    > 「快點！時間來不及了，這件事非常嚴重，我們必須馬上處理，不能再拖了！」
    > *(語速快、用力、緊張)*

*   **想要「專業感」(Scene 2-3) ⮕ 錄這句：**
    > 「這個系統採用了最新的神經網絡架構，數據精確度達到百分之九十九，運行非常穩定。」
    > *(語速穩、清晰、冷靜)*

*   **想要「溫暖感」(Scene 4-5) ⮕ 錄這句：**
    > 「沒關係，慢慢來。我們會一直在這裡陪著你，不用擔心，一切都會好起來的。」
    > *(語速慢、輕柔、像對老人說話)*

### 2. AI Studio 操作步驟
1.  在 **Prompt (提示詞)** 區域，尋找 **"Add audio"** 或 **"Upload"** 按鈕。
2.  上傳您剛剛錄的那段「中文錄音檔」。
3.  在 System Instructions (系統指令) 中加入這行：
    > `Style Reference: Mimic the emotion and pacing of the uploaded audio, but speak the English text provided below.`

---

## 🛠️ Google AI Studio 設定指南 (Setup Guide)

請對照您的截圖畫面進行設定：

1.  **Voice (聲音)**: 推薦選擇 **"Enceladus"** (Deep, Professional) 或 **"Puck"** (Clear, Narrative)。
2.  **Mode (模式)**: 選擇 **Single-speaker audio**。
3.  **操作方式**:
    *   看到 **【複製到 Style instructions】** 的內容，請貼到畫面 **上方** 的格子。
    *   看到 **【複製到 Text】** 的內容，請貼到畫面 **下方** 的格子。

---

## 🛠️ Google AI Studio Setup (System Instruction)

**Copy and paste this into the "System Instructions" block in Google AI Studio to set the persona:**

```text
You are an expert voiceover artist recording a narration for a tech competition video (Google Impact Challenge). 
Your voice is:
1.  **Professional & Authoritative:** Like a senior engineer explaining a critical system.
2.  **Clear & Articulate:** Every word must be distinct.
3.  **Paced:** slightly slower than conversation (0.9x speed) to allow for technical comprehension.
4.  **Empathetic:** When discussing patients, soften your tone.
5.  **Steady:** Maintain a consistent volume and rhythm.

**Pronunciation Rules:**
- "MedGemma": Pronounce as "Med-JEM-mah" (Soft G like 'Gem').
- "SigLIP": Pronounce as "Sig-Lip" (全部唸出, NOT S-I-G-L-I-P).
- "RAG": Pronounce as "Rag" (單字, NOT R-A-G letters).
- "GPU": Pronounce as letters "G-P-U".
- "RAG": Pronounce as "Rag" (rhymes with Bag).
- "CDS": Pronounce as letters "C-D-S".
```

---

## 🎙️ Scene 1: The Hook (0:00 - 0:20)
**BGM: Pulse/Heartbeat (Cyberpunk Style)**

**【複製到 Style instructions】(上方格子)**
> **Speak with high urgency and intensity. Every second counts. You are pitching a life-saving technology.**
> [Tone: Urgent, High Stakes, Fast Paced] (Matching Veo 3.1 Visuals)

**【複製到 Text】(下方格子)**
```text
As an Energy Engineering student, [Proudly] I am trained to prevent system failures.
But in healthcare... errors can have serious consequences.
Existing OCR makes mistakes. Standard LLMs hallucinate.
This is **Silver Guard** C-D-S. A Clinical Decision Support prototype.
```

---

## 🎙️ Scene 2: Edge AI Solution (0:20 - 0:35)
**BGM: Glitch/Suspense -> Silence**

**【複製到 Style instructions】(上方格子)**
> **Start with sudden confusion and alarm ("Wait..."), then switch immediately to confident, reassuring technical authority.**
> [Tone: Dramatic Realization -> Reassuring Expert] (Matching Screen Recording)

**【複製到 Text】(下方格子)**
```text
Wait... [Surprised] Network lost?
*(SFX: Glitch / Static Noise)*
No problem. Watch the privacy shield activate.
Switching to **Air-Gapped Mode**. Zero data leaves this device.
*(Visual: UI turns Grey/Green with "OFFLINE MODE" badge)*
*(Technical Note: Apply "Radio EQ" effect to the line above in post-prod)*
```

---

## 🎙️ Scene 3: Agentic Core (0:35 - 1:25)
**BGM: Tech Minimal (Clean & Rhythm)**

**【複製到 Style instructions】(上方格子)**
> **Speak in a fast, punchy rhythm for the tech stack (SigLIP/MedASR). Then, PAUSE and speak slowly and dramatically for the "Strategy Shift".**
> [Tone: Machine-Gun Fire -> Dramatic Reveal] (Matching Screen Recording)

**【複製到 Text】(下方格子)**
```text
Standard AI guesses. **Silver Guard** validates.
**SigLIP** eyes see "Aspirin". **MedASR** ears hear "Bleeding".
**CRITICAL ALERT: CONTRAINDICATION.**
It uses **Hybrid Verification**... combining VLM reasoning with safety knowledge graphs.
The production system integrates full vector databases for clinical guidelines.
Look at the log... [Pause] **Strategy Shift**.
It detected a risk, **lowered its temperature**, and re-evaluated the logic.
It caught the error that others missed.

We tested this against our **Gallery of Horrors**—synthetic images with extreme physics-based noise.
The result? **Behavioral Stability.**
When the input is garbage, SilverGuard uses its **Input Gate** to refuse safely, rather than hallucinating a dangerous answer.

We trade latency for safety. 
Because getting an answer three seconds late... is better than getting a wrong answer instantly.
```

**🎬 Visual Cue (Tech Stack Overlay):**
*   **0:39 ("SigLIP eyes...")**: Highlight text **"Aspirin 100mg"** on drug bag image.
*   **0:42 ("MedASR ears...")**: Show Audio Waveform + Subtitle: **"Grandma fell and is bleeding now!"**.
*   **0:43 ("Critical Alert")**: **切換到 Gradio 右側 Status Panel** - 背景變粉紅色 (#FFEBEE),顯示 `⛔ HIGH RISK: BLEEDING + ASPIRIN`。**註**: 不是圖片蓋章,是 UI 卡片。
*   **0:45 ("Strategy Shift")**:特寫 (Close-up) the terminal/logs showing "STRATEGY SHIFT: Lowering Temperature -> System 2 Mode" to prove Agentic behavior.

---

## 🎙️ Scene 4: SilverGuard Impact (1:25 - 2:20)
**BGM: Warm Piano/Strings (Emotional)**

**【複製到 Style instructions】(上方格子)**
> **Speak slowly and gently, with deep empathy and warmth. Like a doctor comforting a patient.**
> [Tone: Cinematic, Storytelling, Slower Pace] (Matching Veo 3.1 Visuals)

**【複製到 Text】(下方格子)**
```text
Some might ask: Why is the interface so complex?
Because Silver Guard uses a "Cockpit and Passenger" design.

The dashboard is the "Cockpit" for the caregiver to monitor safety.
The patient never sees this complexity.
They only see what matters: a large-font calendar on the fridge, and a voice alert they can understand.

Raw J-S-O-N is useless to a grandmother. 
Silver Guard translates safety alerts into large-font visuals.

But clarity isn't just for the elderly; it's for those who care for them.
Taiwan's two-hundred-fifty-thousand migrant caregivers now have safety alerts in their language.

Visual safety alerts. Ensuring care has no language barrier.

(Note: This demo uses cloud TTS for audio quality. Production supports offline TTS for strict privacy, with a trade-off in voice naturalness.)
```

[Action: Cross-Lingual Morphing]
Maintain the exact timbre of "Enceladus" but switch language to Indonesian smoothly.
*Fallback: If AI struggles with Indonesian, use Google Translate audio or keep English narration.*
Text: "MOHON TANYA APOTEKER"

---

## 🎙️ Scene 5: Conclusion (2:20 - 2:42)

**【複製到 Style instructions】(上方格子)**
> Speak in an inspirational and resolute tone. This is the final message. End with strength.

**【複製到 Text】(下方格子)**
```text
We are not replacing pharmacists. 
We are giving them a second pair of eyes that never gets tired.

Every alert requires pharmacist verification.
Because clinical decisions always need human judgment.

Powered by **Med-JEM-ma**. Built for privacy. Designed for impact.

This is **Silver Guard** C-D-S. 
Safe. Scalable. And available now on Kaggle.
```

---

## ✅ 製作檢查清單 (Production Checklist)
- [ ] **Voice**: 確認選用了 Enceladus 或 Puck。
- [ ] **Pronunciation**: 試聽 "MedGemma" (唸作 Med-JEM-ma) 和 "SigLIP" (唸作 Sig-Lip) 是否正確。
- [ ] **Slight Pause**: 如果覺得某些地方唸太快，可以在 Text 框裡加入 `...` 或 `[pause]`。
- [ ] **Export**: 生成滿意後，請下載為 WAV 檔。

---

**🏆 預祝錄影順利！**
