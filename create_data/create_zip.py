import random
import re
from collections import defaultdict

zip_templates_mixed = [
    "The package was delivered to zip code {} without any delay.",
    "Please verify the postal code {}, it seems incorrect.",
    "His current address uses the zip code {}.",
    "She recently moved to a new area with postal code {}.",
    "Can you provide the postal code for {}?",
    "All deliveries in that district go through zip code {}.",
    "They registered the business at postal code {}.",
    "Make sure the shipping label shows zip code {}.",
    "I think the postal code for that building is {}.",
    "Residents in zip code {} have reported mail issues.",
    "He’s been living in postal code {} for a decade now.",
    "The zip code {} is known for fast delivery services.",
    "Your postal code {} is not within our service area.",
    "Is postal code {} part of the urban zone?",
    "You wrote the wrong postal code – it should be {}.",
    "We only serve customers in zip code {}.",
    "Postal code {} falls under a restricted delivery zone.",
    "Have you updated your address to postal code {}?",
    "Check again, postal code for that place is {}.",
    "There’s a distribution center near zip code {}."
]


def find_zip_mentions(text):
    patterns = [
        r"\bzip code\s+\d{4}\b",
        r"\bpostal code\s+\d{4}\b",
        r"\bpostal code for\s+\d{4}\b"
    ]
    results = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            mention = m.group()
            results.append((float(m.start()), float(m.end()), mention.strip()))
    return results

def generate_zip_data(filename1, filename2, start_sid=40000, total=500):
    task1, task2 = [], []
    sid = start_sid
    used_sentences = set()

    while len(task1) < total:
        zip_code = random.randint(1000, 9999)
        template = random.choice(zip_templates_mixed)
        sentence = template.format(zip_code)

        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)

        task1.append(f"{sid}\t{sentence}")
        found = False
        for start, end, phrase in find_zip_mentions(sentence):
            if str(zip_code) in phrase:
                task2.append(f"{sid}\tZIP\t{start:.1f}\t{end:.1f}\t{zip_code}")
                found = True

        if not found:
            match = re.search(rf"\b{zip_code}\b", sentence)
            if match:
                task2.append(f"{sid}\tZIP\t{float(match.start()):.1f}\t{float(match.end()):.1f}\t{zip_code}")

        sid += 1

    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f"混合語氣 ZIP 資料已產生，共 {len(task1)} 筆句子與 {len(task2)} 筆標註")


generate_zip_data("task1_zip.txt", "task2_zip.txt")
