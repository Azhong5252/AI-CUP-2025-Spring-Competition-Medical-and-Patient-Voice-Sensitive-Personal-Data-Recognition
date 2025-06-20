# 讀取原始檔案
with open("GPT.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 篩選出只包含 SHI TYPE 為 DATE 的行
date_lines = [line for line in lines if "\tDOCTOR\t" in line]  # 確保前後是制表符

# 確認篩選結果
print(f"篩選出的行數: {len(date_lines)}")
if date_lines:
    print("篩選到的範例:")
    print(date_lines[:5])  # 打印前5行來檢查篩選結果

# 複寫回檔案
if date_lines:
    with open("1.txt", "w", encoding="utf-8") as f:
        f.writelines(date_lines)
else:
    print("沒有找到任何符合條件的行，檢查標籤格式。")
