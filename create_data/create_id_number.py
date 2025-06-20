import random
import re

id_number_templates = [

    "This medical report pertains to Addison Dado, identified with ID number {}, medical record 634741.LNA, and lab number.",
    "This report concerns Harley Rader, medical record number 057758.kbp, ID number {}, and lab number 05T7830.",
    "This medical case pertains to Virginia James, whose medical record number is 664499.seu. Her lab number is 66M49971, and her documented medical history includes multiple significant dates. Her ID number is {}.",
    "Please verify patient identity using ID number {}.",
    "Subject ID number is {} was flagged during the review process.",
    "Archived under ID number {} for audit purposes.",
    "All visit records are tied to ID number {}.",
    "Patient carries ID number is {} on all official documents.",
    "A duplicate was found for ID number {}.",
    "The system logged in under ID number is {}.",
    
    "Patient's lab number is {}.",
    "The specimen was tagged with lab number {}.",
    "Lab number {} was used to identify the blood sample.",
    "Test results are stored under lab number {}.",
    "The lab record, marked as lab number {}, was archived.",
    "Sample analysis completed for lab number {}.",
    "Biopsy was registered under lab number {}.",
    "Blood test is linked to lab number {}.",
    "Lab number {} was flagged for reanalysis.",
    "A second opinion was requested for lab number {}.",
]


def generate_id_number():
    formats = [
        lambda: f"{random.randint(10,99)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000,999999)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",  # e.g. 12R423044B
        lambda: f"{random.randint(10,99)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000,999999)}",  # e.g. 63M741450
        lambda: f"{random.randint(10,99)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000,9999)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",  # short variant
        lambda: f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1,9)}",  # e.g. B1
    ]
    return random.choice(formats)()

def find_id_number_mentions(text):
    pattern = r"\b(?:ID number|lab number)\s+(\d{2}[A-Z]\d{4,6}[A-Z]?|[A-Z]\d)\b"
    results = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        start, end = m.start(1), m.end(1)
        mention = m.group(1)
        results.append((float(start), float(end), mention.strip()))
    return results


def generate_id_number_data(filename1, filename2, start_sid=50000, total=500):
    task1, task2 = [], []
    sid = start_sid
    used_sentences = set()

    while len(task1) < total:
        id_number = generate_id_number()
        template = random.choice(id_number_templates)
        sentence = template.format(id_number)

        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)

        task1.append(f"{sid}\t{sentence}")
        found = False
        for start, end, phrase in find_id_number_mentions(sentence):
            if id_number in phrase:
                task2.append(f"{sid}\tID_NUMBER\t{start:.1f}\t{end:.1f}\t{phrase}")
                found = True

        if not found:
            match = re.search(re.escape(id_number), sentence)
            if match:
                task2.append(f"{sid}\tID_NUMBER\t{float(match.start()):.1f}\t{float(match.end()):.1f}\t{id_number}")

        sid += 1

    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f" ID_NUMBER 資料已產生，共 {len(task1)} 筆句子與 {len(task2)} 筆標註")

generate_id_number_data("task1_id_number.txt", "task2_id_number.txt")
