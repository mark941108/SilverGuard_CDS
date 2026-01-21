# 🎬 MedGemma Impact Challenge - 冠軍影片製作指南
## AI Pharmacist Guardian Demo Video Script (2:50)

---

## 📋 影片製作清單

- [ ] **OBS Studio 安裝與設定**
- [ ] **錄製素材** (Cell 5 Terminal + Cell 7 SilverGuard)
- [ ] **AI 語音生成** (ElevenLabs)
- [ ] **剪輯合成** (CapCut/DaVinci)
- [ ] **上傳 YouTube** (Unlisted)

---

## 🎯 影片核心策略

> **前 10 秒抓住注意力 → 中間 60 秒展示 Agentic → 最後 30 秒用 SilverGuard 收尾**

評審看幾百支影片，你的影片必須在前 10 秒讓他們決定要認真看。

---

## 📝 完整分鏡腳本

### Scene 1: The Hook (0:00 - 0:20) — 恐懼與痛點

| 時間 | 畫面 | 旁白 (英文) |
|------|------|-------------|
| 0:00-0:05 | Hero Image 滿版 + 標題動畫 | *"I am an **Energy Engineering student**, not a doctor."* |
| 0:05-0:10 | 疊加文字: **"System Stability"**, **"Fail-Safe"** | *"In my field, 'system stability' is everything."* |
| 0:10-0:15 | 模糊藥袋照片 (你的合成數據) | *"So when I saw how fragile medication safety is for the elderly..."* |
| 0:15-0:20 | 紅色警告畫面 + **"⚠️ LETHAL RISK"** | *"...I decided to build a system that doesn't just read text—it protects life."* |

**🎬 製作提示:**
- 用快節奏剪輯
- 紅色文字強調數據
- 背景音樂: 緊張氛圍 (10% 音量)

---

### Scene 2: The Solution (0:20 - 0:50) — Edge AI 價值

| 時間 | 畫面 | 旁白 (英文) |
|------|------|-------------|
| 0:20-0:30 | Kaggle Notebook 介面 + T4 GPU 標籤 + **"MedGemma 1.5"** Logo | *"Meet AI Pharmacist Guardian. Core inference runs 100% locally on a single T4 GPU."* |
| 0:30-0:40 | 圖卡: **"Zero PII Egress"**, **"Hybrid Privacy"**, **"HIPAA-Compliant"** | *"Image analysis and PHI extraction happen on-device. Only anonymized drug names query external safety DBs in Hybrid Mode."* |
| 0:40-0:50 | 你的架構 Mermaid 圖 (Animated) | *"This is not just OCR. It's an Agentic Workflow with self-correction capability, built on the Gemma 3 architecture."* |

**🎬 製作提示:**
- **戰略關鍵字植入**: 畫面必須出現 "SigLIP" 和 "Gemma 3 Architecture" 字樣 (Google 評審愛看)。
- 強調 **Privacy** 和 **Edge**。
- 圖卡使用高對比色彩。

---

### Scene 3: The Magic - Agentic Loop (0:50 - 1:50) — 技術核心 ⚠️ 最重要

**關鍵策略：拒絕無聊的 Terminal！將「思考過程」視覺化 (Visualize the Thinking)**

| 時間 | 畫面 | 旁白 (英文) |
|------|------|-------------|
| 0:50-1:05 | **[畫面分割]** 左邊是模糊藥袋圖 (Noisy Data)，右邊是 Agent | *"Watch as we feed it a **'Nightmare Case'**—blurry, dark, and confusing. A standard model would fail here."* |
| 1:05-1:15 | **[特效]** `Logic Flaw Detected` 出現時畫面**凍結** 👉 **出現「大腦思考」動畫** | *"But my Agent acts like a human pharmacist. It triggers a **Self-Correction Loop**, realizing the dosage is lethal for an 88-year-old."* |
| 1:15-1:25 | **[動畫]** 顯示 Prompt 文字變化: `Adding Context: Please re-analyze...` | *"It doesn't just guess; it **reasons, reflects, and retries**."* |
| 1:25-1:35 | **[特效]** 顯示溫度計動畫: **Temperature 0.6 ➔ 0.2** 🌡️ | *"It reduces temperature to 0.2 for deterministic reasoning, and retries."* |
| 1:25-1:35 | **[特效]** 顯示溫度計動畫: **Temperature 0.6 ➔ 0.2** 🌡️ | *"It reduces temperature to 0.2 for deterministic reasoning, and retries."* |
| 1:35-1:40 | **[左右對比]** 左邊: Attempt 1 (❌ Fail) vs 右邊: Attempt 2 (✅ Success) | *"And there it is—HIGH_RISK correctly flagged. This is true Agentic Intelligence."* |
| 1:40-1:50 | **[終極對比]** 分割畫面：左邊 Standard OCR (亂碼/錯誤) vs 右邊 MedGemma Agent (拒絕/修正) | *"While standard OCR blindly accepts errors, our Agent refuses to guess. A refusal is safer than a hallucination."* |

**🎬 製作提示:**
- **絕對不要只錄 Terminal 滾動！**
- **Overlay (覆蓋層)**: 在 Terminal 上面蓋一個半透明圖層，用大字寫 **"Step 1: Self-Reflection Triggered"**。
- **證據展示**: 務必用那張「有噪點、爛爛的」合成圖，不要用完美的圖，證明你的 Robustness。
- **溫度細節**: 把 `Temperature 0.6 -> 0.2` 做成一個小動畫，這是技術亮點。

---

### Scene 4: Social Impact - SilverGuard & Health Equity (1:50 - 2:30) — 情感收尾

| 時間 | 畫面 | 旁白 (英文) |
|------|------|-------------|
| 1:50-2:00 | 左右對比: JSON 輸出 (Confusing) vs SilverGuard (Clear) | *"Complex JSON is meaningless to elderly patients. That's why we built SilverGuard."* |
| 2:00-2:10 | SilverGuard 大字體日曆 (HTML 截圖) | *"Large 28px+ fonts, high-contrast colors designed for WCAG 2.1 AAA accessibility..."* |
| 2:10-2:20 | **[Demo]** 滑鼠切換語言至 **Indonesian**，警告轉為 **"BAHAYA!"** | *"But we didn't stop there. For Taiwan's 250,000 migrant caregivers, we added a **Language Bridge**."* |
| 2:20-2:30 | **播放 Indonesian TTS 語音** + 阿嬤與外籍看護微笑圖 | *"Verified medical alerts in Indonesian and Vietnamese. Ensuring safety has no language barrier."* |

**🎬 製作提示:**
- **聽覺陷阱**: TTS 語音必須夠「台」、夠溫暖。並展示印尼語音的 **"BAHAYA!" (危險)** 震撼力。
- **字幕戰術**: 評審聽不懂中文/印尼文，**一定加英文字幕**： *(Subtitle: "DANGER! Do not take this!")*。
- 背景音樂轉為溫馨感人。

---

### Scene 5: Call to Action (2:30 - 2:50) — 乾淨收尾

| 時間 | 畫面 | 旁白 (英文) |
|------|------|-------------|
| 2:30-2:40 | Hero Image + GitHub/Kaggle 連結 | *"Built by **one student**, running on **free hardware**, designed for **every forgotten elder**."* |
| 2:40-2:50 | 標語: **"AI Pharmacist Guardian: Safety First, Privacy Always."** | *"This is AI Pharmacist Guardian: Turning personal regret into public safety."* |

---

## 🛠️ 製作工具設定

### 1. OBS Studio 錄製設定 (來源: 2024 最佳實踐)

| 設定項目 | 推薦值 | 說明 |
|----------|--------|------|
| **解析度** | 1920x1080 | 1080p 足夠，不需要 4K |
| **幀率** | 30 FPS | 足夠流暢，檔案較小 |
| **格式** | MKV | 防止當機時檔案損壞，後轉 MP4 |
| **編碼器** | NVENC H.264 | 使用 GPU 硬體編碼 |
| **Rate Control** | CQP | 品質優先 |
| **CQ Level** | 18-20 | 高品質平衡檔案大小 |

**錄製技巧:**
1. 錄製「視窗」而非全螢幕
2. **放大字體 (Ctrl +)** 再錄，確保手機也看得清楚
3. 把滑鼠游標調大或加黃色光圈

### 2. ElevenLabs AI 語音設定 (來源: 專業語音最佳實踐)

| 設定項目 | 推薦值 |
|----------|--------|
| **語音** | Rachel / Adam (專業旁白風格) |
| **穩定性** | 70-80% |
| **清晰度** | 80-90% |
| **速度** | 0.9x (稍慢，更清楚) |

**操作步驟:**
1. 前往 https://elevenlabs.io
2. 選擇 "Text to Speech"
3. 貼入上面的英文旁白腳本
4. 選擇 "Rachel" 語音 (專業女聲) 或 "Adam" (專業男聲)
5. 下載 MP3

### 3. CapCut 剪輯技巧

**必用功能:**
- **Zoom (縮放)**: 當顯示 `Logic Flaw Detected` 時，畫面縮放到那行字
- **文字特效**: 疊加關鍵數據 (例如 "$42B", "7x Risk")
- **Speed Up (加速)**: 把模型載入等待時間加速 8x
- **Audio Ducking**: 旁白說話時，BGM 自動降低

---

## 📜 完整英文旁白腳本 (複製貼上到 ElevenLabs)

```text
[Scene 1: Hook - 0:00-0:30]
This could kill an 88-year-old. But standard AI thinks it's safe. In 2025, as Taiwan enters a 'Super-Aged Society', we need something better than a chatbot. We need a Guardian.
I'm [Your Name], and I built an Agent that catches what Google and GPT-4 miss.

[Scene 2: Solution - 0:30-1:00]
Meet AI Pharmacist Guardian.
Powered by Google's MedGemma 1.5, running 100% locally on a single T4 GPU.
This is our "Privacy Moat": Unlike cloud APIs that leak sensitive PHI, our Edge AI architecture keeps patient data logically isolated and HIPAA-compliant.
It utilizes a Hybrid Architecture: Neural perception for reading, Symbolic logic for safety.

[Scene 3: Agentic Loop - 1:00-2:00]
Real-world data is messy. Standard models hallucinate on blur. So we stress-tested our Agent against a 'Gallery of Horrors'.
Watch as we feed it a 'Nightmare Case'—blurry, dark, and confusing. A standard model would fail here.
But my Agent acts like a human pharmacist. It triggers a Self-Correction Loop, realizing the dosage is lethal for an 88-year-old.
It doesn't just guess; it reasons, reflects, and retries.
It reduces temperature to 0.2 for deterministic reasoning, and retries.
And there it is—HIGH_RISK correctly flagged. This is true Agentic Intelligence: observe, evaluate, adapt, retry.
And when it's not sure? It doesn't guess. It calls a human. Safety is knowing when to stop.

[Scene 4: SilverGuard & Health Equity - 2:00-2:40]
Complex JSON saves no one. SilverGuard translates code into care. Large fonts, WCAG accessibility, and distinct audio alerts.
And for the 250,000 migrant caregivers in Taiwan, we built a Language Bridge. Safety in their native tongue.
We don't just detect errors—we bridge the communication gap for the most vulnerable.

[Scene 5: Call to Action - 2:40-2:50]
Built by one student, running on free hardware, designed for every forgotten elder. This is AI Pharmacist Guardian: Turning personal regret into public safety.
```

---

## ⏱️ 時間控制檢查

| 段落 | 目標時長 | 實際 |
|------|----------|------|
| Hook (恐懼) | 20 秒 | __ |
| Solution (Edge AI) | 30 秒 | __ |
| Agentic Loop (核心) | 60 秒 | __ |
| SilverGuard (情感) | 40 秒 | __ |
| Call to Action | 20 秒 | __ |
| **總計** | **2:50** | __ |

> ⚠️ **警告**: 絕對不能超過 3:00！規則明確規定 "3 minutes or less"

---

## ✅ 最終檢查清單

- [ ] 影片長度 ≤ 2:59
- [ ] **MedGemma** 字樣有出現 (給 Google 面子)
- [ ] **AGS Beers Criteria** 有提到 (醫學權威性)
- [ ] Agentic Loop 的 Log 有放大顯示
- [ ] TTS 語音有播放出來
- [ ] 字幕清楚可讀 (手機也看得懂)
- [ ] 上傳 YouTube (Unlisted)
- [ ] 連結貼到 Writeup

---

**🏆 做完這支影片，你的 Agentic Workflow Prize 就穩了！**
