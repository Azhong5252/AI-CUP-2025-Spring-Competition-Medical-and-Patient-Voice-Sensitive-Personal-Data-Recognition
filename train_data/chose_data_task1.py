# 读取task2_answer.txt中的编号
def read_task2_file(task2_file):
    task2_ids = set()  # 使用set去重编号
    with open(task2_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            task2_ids.add(parts[0].strip())  # 去除编号的前后空白字符
    return task2_ids

# 从task1_answer.txt中提取相应的行
def extract_task1_lines(task1_file, task2_ids, output_file):
    with open(task1_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        for line in f:
            task1_id = line.split()[0].strip()  # 获取task1的编号并去掉空格
            if task1_id in task2_ids:
                out_f.write(line)  # 如果编号匹配，写入输出文件

# 主函数
def main():
    task1_file = 'task1_answer.txt'  # task1_answer.txt 文件路径
    task2_file = 'task2_date.txt'  # task2_answer.txt 文件路径
    output_file = 'task1_date.txt'    # 输出文件路径

    # 读取task2_answer.txt中的编号
    task2_ids = read_task2_file(task2_file)
    print(f"Extracted {len(task2_ids)} unique IDs from task2_answer.txt")  # 打印读取到的ID数量

    # 提取task1_answer.txt中的对应行并写入task_data.txt
    extract_task1_lines(task1_file, task2_ids, output_file)

    print(f"Task data has been written to {output_file}")

# 运行主程序
if __name__ == '__main__':
    main()