def run():
    # 強化版讀取 & 排序 & 輸出
    with open("validation/inference_output.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    def safe_key(line):
        try:
            return int(line.split('\t')[0])
        except Exception:
            return float('inf')

    # 確保每一行至少5個欄位（編號、標籤、開始時間、結束時間、文字）
    fixed_lines = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 3:
            parts += [""] * (5 - len(parts))  # 不夠的補空白
        fixed_lines.append('\t'.join(parts))

    sorted_lines = sorted(fixed_lines, key=safe_key)

    with open("validation/inference_output.txt", "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(line + '\n')

    print("✅ 完成排序並補齊欄位！")
if __name__ == "__main__":
    run()