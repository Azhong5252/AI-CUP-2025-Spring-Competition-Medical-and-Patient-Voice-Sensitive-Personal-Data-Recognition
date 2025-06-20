def run():
    with open("validation/inference_output.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    def safe_key(line):
        try:
            return int(line.split('\t')[0])
        except Exception:
            return float('inf')

    fixed_lines = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 3:
            parts += [""] * (5 - len(parts))  
        fixed_lines.append('\t'.join(parts))

    sorted_lines = sorted(fixed_lines, key=safe_key)

    with open("validation/inference_output.txt", "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(line + '\n')

    print("完成排序並補齊欄位！")
if __name__ == "__main__":
    run()