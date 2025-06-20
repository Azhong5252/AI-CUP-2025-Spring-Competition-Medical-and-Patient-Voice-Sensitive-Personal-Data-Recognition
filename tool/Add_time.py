# import json
# import re

# def run():
#     timestamp_file = "ASR_code/text/Whisper_Validation_Timestamps.json"
#     gpt_file = "validation/inference_output.txt"

#     with open(timestamp_file, "r", encoding="utf-8") as f:
#         timestamps = json.load(f)

#     def normalize_text(text):
#         return re.sub(r"[^\w]", "", text).lower()

#     def find_matching_sequence(word_list, target_words):
#         target_norm = "".join(normalize_text(w) for w in target_words)

#         for i in range(len(word_list)):
#             combined = ""
#             for j in range(i, len(word_list)):
#                 word_norm = normalize_text(word_list[j]["word"])
#                 combined += word_norm
#                 if combined == target_norm:
#                     return word_list[i]["start"], word_list[j]["end"]
#                 if not target_norm.startswith(combined):
#                     break

#         for i in range(len(word_list)):
#             for word in target_words:
#                 if normalize_text(word) in normalize_text(word_list[i]["word"]):
#                     return word_list[i]["start"], word_list[i]["end"]
#         return None, None

#     with open(gpt_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     updated_lines = []
#     removed_count = 0  # ➕ 統計被刪除的行數（可選）
#     for line in lines:
#         parts = line.strip().split(None, 2)
#         if len(parts) < 3:
#             updated_lines.append(line.strip())
#             continue

#         utt_id, entity_type, entity_text = parts[0], parts[1], parts[2]

#         # 🧹 直接刪除 entity_text == entity_type（大小寫不敏感）的行
#         if entity_text.strip().lower() == entity_type.lower():
#             removed_count += 1
#             continue

#         if utt_id not in timestamps:
#             updated_lines.append(line.strip())
#             continue

#         word_list = timestamps[utt_id]

#         # ✅ 對 DOCTOR 處理 Dr. 和 Drs. 前綴
#         if entity_type == "DOCTOR":
#             entity_text = re.sub(r"^\s*Dr\.?\s+", "", entity_text, flags=re.IGNORECASE)
#             entity_text = re.sub(r"^\s*Drs\.?\s+", "", entity_text, flags=re.IGNORECASE)

#         words = entity_text.strip().split()
#         start_time, end_time = find_matching_sequence(word_list, words)

#         if start_time is not None and end_time is not None:
#             updated_line = f"{utt_id} {entity_type} {start_time} {end_time} {entity_text}"
#         else:
#             updated_line = f"{utt_id} {entity_type} {entity_text}"

#         updated_lines.append(updated_line)

#     with open(gpt_file, "w", encoding="utf-8") as f:
#         f.write("\n".join(updated_lines))

#     print(f"✅ 完成：共刪除 {removed_count} 行 entity_text == entity_type 的資料")
# if __name__ == "__main__":
#     run()


import json
import re

def run():
    timestamp_file = "ASR_code/text/Whisper_Validation_Timestamps.json"
    gpt_file = "validation/inference_output.txt"

    with open(timestamp_file, "r", encoding="utf-8") as f:
        timestamps = json.load(f)

    def normalize_text(text):
        return re.sub(r"[^\w]", "", text).lower()

    def find_matching_sequence(word_list, target_words, used_indices):
        target_norm = "".join(normalize_text(w) for w in target_words)

        for i in range(len(word_list)):
            if i in used_indices:
                continue  # 跳過已用過的詞

            combined = ""
            indices_in_combo = []
            for j in range(i, len(word_list)):
                if j in used_indices:
                    break  # 中間詞已用過，無法連續匹配

                word_norm = normalize_text(word_list[j]["word"])
                combined += word_norm
                indices_in_combo.append(j)

                if combined == target_norm:
                    # 找到完整匹配，標記這些詞已用過
                    used_indices.update(indices_in_combo)
                    return word_list[i]["start"], word_list[j]["end"]
                if not target_norm.startswith(combined):
                    break

        # 多詞組合找不到，改找單字匹配（先跳過用過的詞）
        for i in range(len(word_list)):
            if i in used_indices:
                continue

            for word in target_words:
                if normalize_text(word) in normalize_text(word_list[i]["word"]):
                    used_indices.add(i)
                    return word_list[i]["start"], word_list[i]["end"]

        return None, None

    with open(gpt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    removed_count = 0

    # 這裡建立一個字典：utt_id -> set(used word indices)
    used_word_indices_map = {}

    for line in lines:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            updated_lines.append(line.strip())
            continue

        utt_id, entity_type, entity_text = parts[0], parts[1], parts[2]

        # 刪除 entity_text == entity_type（不分大小寫）
        if entity_text.strip().lower() == entity_type.lower():
            removed_count += 1
            continue

        if utt_id not in timestamps:
            updated_lines.append(line.strip())
            continue

        word_list = timestamps[utt_id]

        if utt_id not in used_word_indices_map:
            used_word_indices_map[utt_id] = set()

        # DOCTOR 特殊處理 Dr.、Drs. 前綴
        if entity_type == "DOCTOR":
            entity_text = re.sub(r"^\s*Dr\.?\s+", "", entity_text, flags=re.IGNORECASE)
            entity_text = re.sub(r"^\s*Drs\.?\s+", "", entity_text, flags=re.IGNORECASE)

        words = entity_text.strip().split()

        start_time, end_time = find_matching_sequence(word_list, words, used_word_indices_map[utt_id])

        if start_time is not None and end_time is not None:
            updated_line = f"{utt_id} {entity_type} {start_time} {end_time} {entity_text}"
        else:
            updated_line = f"{utt_id} {entity_type} {entity_text}"

        updated_lines.append(updated_line)

    with open(gpt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines))

    print(f"✅ 完成：共刪除 {removed_count} 行 entity_text == entity_type 的資料")

if __name__ == "__main__":
    run()
