import random
import re

# SET 相關樣板句
set_templates = [
    "The medication was prescribed to be taken {}.",
    "Patient is instructed to do physiotherapy {}.",
    "She takes her pills {} without fail.",
    "Doctor recommended light exercise {}.",
    "Apply the cream {} to affected area.",
    "Monitor your blood sugar {}.",
    "He needs dialysis {} as per medical advice.",
    "Vitals should be recorded {}.",
    "I usually take the supplements {}.",
    "They advised me to check my blood pressure {}.",
    "The injection is scheduled {} until symptoms improve.",
    "Use the inhaler {} to prevent asthma attacks.",
    "She visits the clinic {} for follow-up.",
    "Follow-up tests are to be done {}.",
    "The patient receives physiotherapy sessions {}.",
    "Medication should be taken {} after meals.",
    "Doctor visits are required {} until discharge.",
    "I go for acupuncture therapy {}.",
    "Instructions say to take the drops {}.",
    "The treatment is administered {} based on progress.",
    
    # 擴充樣板
    "He performs breathing exercises {} as part of his routine.",
    "The pills must be taken {} regardless of food intake.",
    "Measurements of oxygen saturation are done {}.",
    "I take this medication {} to manage blood pressure.",
    "Patient reports taking herbal supplements {}.",
    "The ointment should be applied {} for best results.",
    "Glucose levels are checked {} with a home monitor.",
    "The nurse gives injections {}.",
    "Our follow-up calls are scheduled {}.",
    "Painkillers should only be taken {}.",
    "He attends rehab sessions {} following his surgery.",
    "They instructed him to use the nebulizer {}.",
    "Appointments will be arranged {} until improvement is noted.",
    "Medication refills are requested {}.",
    "Patient performs stretching exercises {}.",
    "Physical therapy continues {} to aid recovery.",
    "The caregiver administers medication {}.",
    "Blood pressure should be measured {} to monitor fluctuations.",
    "We ask patients to complete questionnaires {}."
]


# SET 表達詞彙列表
set_phrases = [
    "daily", "once daily", "twice daily", "three times daily",
    "once a day", "twice a day", "three times a day",
    "once a week", "twice a week", "three times a week",
    "weekly", "biweekly", "every morning", "every evening", "every night",
    "every 8 hours", "every 12 hours", "every 6 hours", "every other day",
    "as needed", "as required", "prn", "before each meal", "after each meal",
    "each night", "every few days", "at bedtime", "before sleep", "in the morning",
    "in the evening", "every hour", "every weekend", "each morning", "every Tuesday"
]


def find_set_mentions(text):
    results = []
    for phrase in set_phrases:
        pattern = re.escape(phrase)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            results.append((float(match.start()), float(match.end()), match.group()))
    return results

def generate_set_data(filename1, filename2, start_sid=40000, total=500):
    task1, task2 = [], []
    sid = start_sid
    used_sentences = set()

    while len(task1) < total:
        phrase = random.choice(set_phrases)
        template = random.choice(set_templates)
        sentence = template.format(phrase)

        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)

        task1.append(f"{sid}\t{sentence}")

        found = False
        for start, end, matched_text in find_set_mentions(sentence):
            if matched_text.lower() == phrase.lower():
                task2.append(f"{sid}\tSET\t{start:.1f}\t{end:.1f}\t{matched_text}")
                found = True

        if not found:
            print(f"❗無法標註句子: {sentence}")

        sid += 1

    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f"✅ SET 類型資料已產生，共 {len(task1)} 筆句子與 {len(task2)} 筆標註")

# 執行
generate_set_data("task1_set.txt", "task2_set.txt")
