import sys
import re
def run():
    input_path = "validation/inference_output.txt"
    output_path = "validation/inference_output.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []
    pattern = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)")

    for line in lines:
        match = pattern.match(line)
        if match:
            fields = match.groups()
            results.append("\t".join(fields))
        else:
            results.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")
    print(f"轉換完成！已輸出到 {output_path}")
if __name__ == "__main__":
    run()
