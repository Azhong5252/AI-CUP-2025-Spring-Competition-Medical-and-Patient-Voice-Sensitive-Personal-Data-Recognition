import random
import re
import string

# MEDICAL_RECORD_NUMBER 專用模板
medical_record_templates = [
    "Patient was identified using medical record {}.",
    "The patient's medical record number is {}.",
    "Access to medical record {} was requested.",
    "Verification completed for medical record number {}.",
    "The form listed medical record {} as reference.",
    "Errors were found in medical record number {} during the audit.",
    "Consent was linked to medical record {}.",
    "Admission papers stated medical record number {}.",
    "The system retrieved details for medical record {}.",
    "Records were updated under medical record number {}.",
    "Medical record {} was flagged for further review.",
    "Please provide your medical record number {} at check-in.",
    "Medical records team confirmed details under {}.",
    "Issues related to medical record {} were resolved last week.",
    "All billing information is tied to medical record number {}.",
    "We found discrepancies in medical record {} during the review.",
    "Your appointment has been linked to medical record number {}.",
    "Data migration was initiated for medical record {}.",
    "Notification sent regarding updates in medical record number {}.",
    "Medical record {} is archived in our system now.",
]


# 生成 medical record 數字（可能加.字母）
def generate_medical_record_number():
    number = str(random.randint(100000, 999999))  # 6位數
    if random.random() < 0.5:
        suffix = '.' + ''.join(random.choices(string.ascii_uppercase, k=3))  # 加 .ABC
        return number + suffix
    else:
        return number

# 抓取 medical record number 出現的地方
def find_medical_record_number_mentions(text):
    pattern = r"(?:medical record(?: number)? )(\d{5,7}(?:\.[A-Z]{2,4})?)"
    results = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = m.start(1)
        end = m.end(1)
        mention = m.group(1)
        results.append((float(start), float(end), mention.strip()))
    return results

# 產生資料並保存
def generate_medical_record_number_data(filename1, filename2, start_sid=40000, total=500):
    task1, task2 = [], []
    sid = start_sid
    used_sentences = set()

    while len(task1) < total:
        record_number = generate_medical_record_number()
        template = random.choice(medical_record_templates)
        sentence = template.format(record_number)

        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)

        task1.append(f"{sid}\t{sentence}")
        found = False
        for start, end, phrase in find_medical_record_number_mentions(sentence):
            if record_number in phrase:
                task2.append(f"{sid}\tMEDICAL_RECORD_NUMBER\t{start:.1f}\t{end:.1f}\t{record_number}")
                found = True

        # fallback：直接搜
        if not found:
            match = re.search(re.escape(record_number), sentence)
            if match:
                task2.append(f"{sid}\tMEDICAL_RECORD_NUMBER\t{float(match.start()):.1f}\t{float(match.end()):.1f}\t{record_number}")

        sid += 1

    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f"✅ MEDICAL_RECORD_NUMBER 資料已產生，共 {len(task1)} 筆句子與 {len(task2)} 筆標註")

# 使用
generate_medical_record_number_data("task1_medical_record.txt", "task2_medical_record.txt")
