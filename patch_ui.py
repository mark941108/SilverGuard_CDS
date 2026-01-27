
import os

target_file = "AI_Pharmacist_Guardian_V5.py"
start_line = 2275
end_line = 2311

new_content = """        if status in ["HIGH_RISK", "PHARMACIST_REVIEW_REQUIRED"]:
            speech = f\"\"\"
⚠️ {patient_name}，系統提醒您留意喔！

這包「{friendly_drug}」上面的劑量寫著 {dose}，
機器人查了一下資料，覺得跟一般老人家用的習慣不太一樣。

👉 為了安全起見，這包藥我們先放旁邊，
麻煩您拿給藥局的哥哥姊姊看一下，確認沒問題我們再吃，好不好？
{disclaimer}
\"\"\"
        elif status in ["WARNING", "ATTENTION_NEEDED"]:
            speech = f\"\"\"
🟡 {patient_name}，要注意喔！

這包「{friendly_drug}」在吃的時候要注意：
{reasoning}

👉 下次看醫生的時候，可以把藥袋帶著，順便問一下醫生這樣吃對不對？
{disclaimer}
\"\"\"
        elif status in ["PASS", "WITHIN_STANDARD"]:
            speech = f\"\"\"
✅ {patient_name}，這包藥沒問題喔！

這是您的「{friendly_drug}」。
吃法：{usage}
劑量：{dose}

記得要吃飯後再吃，才不會傷胃喔！身體會越來越健康的！
{disclaimer}
\"\"\"
        else:
            speech = f\"\"\"
⚠️ {patient_name}，AI 不太確定這張照片。

👉 建議：請拿藥袋直接問藥師比較安全喔！
{disclaimer}
\"\"\"
"""

with open(target_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Python list is 0-indexed, lines are 1-indexed
# We want to replace lines[start_line-1 : end_line]
# slice is start_index inclusive, end_index exclusive.
# line 2275 is index 2274.
# line 2311 is index 2310.
# we want to include 2311 in deletion. So end index is 2311.
lines[start_line-1 : end_line] = [new_content]

with open(target_file, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ Patch applied successfully.")
