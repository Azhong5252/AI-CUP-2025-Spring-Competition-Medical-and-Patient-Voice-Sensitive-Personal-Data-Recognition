import random
import re
from collections import defaultdict
from num2words import num2words

age_templates_mixed = [
    "The {}-year-old patient presented with symptoms consistent with pneumonia.",
    "Medical records indicate the subject was {} years old at the time of admission.",
    "A comprehensive health check was conducted when the patient turned {}.",
    "At the age of {}, the patient was diagnosed with type 2 diabetes.",
    "Reports suggest better recovery rates in patients aged {} and above.",
    "She had her annual physical exam at {}.",
    "I remember back when I was {}, things were so different.",
    "He's already {}? Time really flies.",
    "She mentioned she's {} during our last phone call.",
    "Didn't he say he was turning {} soon?",
    "When you reach {}, you start valuing health more.",
    "She hit {} last month and threw a huge party.",
    "After turning {}, he noticed changes in his endurance levels.",
    "Being in her {}, she maintains an active lifestyle.",
    "By the time he was {}, he had already traveled the world.",
    "At {}, managing diet becomes crucial for health.",
    "Even at {}, she runs marathons regularly.",
    "Patients over {} require closer monitoring post-surgery.",
    "The risk assessment was adjusted for individuals aged {}.",
    "He celebrated his {}th birthday surrounded by friends."
]

def number_to_words(n):
    return num2words(n).replace("-", " ")

def find_age_mentions(text):
    patterns = [
        r"\b\d{1,3}\s+years old",
        r"\b\d{1,3}-year-old",
        r"\bat the age of\s+\d{1,3}",
        r"\baged\s+\d{1,3}",
        r"\bat\s+\d{1,3}",
        r"\bturning\s+\d{1,3}",
        r"\b(?:he|she|they|i|we)\s+(?:was|is|'s|were|are|am|had been|been)\s+\d{1,3}",
        r"\b\d{1,3}(?:th)?\s+birthday",
        r"\bin\s+(?:his|her|their)\s+\d{2}s",
        r"\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|[a-z\-]+)\s+year-old",
        r"\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|[a-z\-]+)\s+years old",
    ]
    results = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            mention = m.group()
            results.append((float(m.start()), float(m.end()), mention.strip()))
    return results

def generate_age_data(filename1, filename2, start_sid=30000, total=500):
    task1, task2 = [], []
    sid = start_sid
    used_sentences = set()

    while len(task1) < total:
        age = random.randint(2, 99)

        if random.random() < 0.3:
            age_text = number_to_words(age)
        else:
            age_text = str(age)

        template = random.choice(age_templates_mixed)
        sentence = template.format(age_text)

        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)

        task1.append(f"{sid}\t{sentence}")
        found = False
        for start, end, phrase in find_age_mentions(sentence):
            if str(age) in phrase or age_text.replace("-", " ") in phrase.lower():
                task2.append(f"{sid}\tAGE\t{start:.1f}\t{end:.1f}\t{age}")
                found = True

        if not found:
            match = re.search(rf"\b{age}\b", sentence)
            if match:
                task2.append(f"{sid}\tAGE\t{float(match.start()):.1f}\t{float(match.end()):.1f}\t{age}")

        sid += 1

    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f"混合語氣 AGE 資料已產生，共 {len(task1)} 筆句子與 {len(task2)} 筆標註")

generate_age_data("task1_age.txt", "task2_age.txt")