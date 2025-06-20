import os
import shutil

def run():
    def filter_file(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                stripped = line.strip()
                if not stripped:
                    continue
                columns = stripped.split('\t')
                if len(columns) == 5:
                    outfile.write(line)

    input_path = 'validation/inference_output.txt'
    output_path = 'validation/inference_output_filtered.txt'

    if not os.path.exists(input_path):
        print(f"找不到檔案：{input_path}")
    else:
        filter_file(input_path, output_path)
        shutil.move(output_path, input_path)  
        print(f"已完成過濾並更新 {input_path}")
            
if __name__ == "__main__":
    run()