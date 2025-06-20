def run():
    input_file = "validation/inference_output.txt"
    output_file = "validation/inference_output.txt"

    seen = set()
    deduped_lines = []

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if line not in seen:
                seen.add(line)
                deduped_lines.append(line)

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(deduped_lines)

    print(f"去除重複完成，結果儲存於 {output_file}")