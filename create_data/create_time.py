import random
import re

# 多句式 + 時間自然融合的模板（含上下文與因果）
time_templates_rich = [
    "She woke up feeling unwell and by {}, she had already taken her meds and gone back to sleep.",
    "The team gathered in the conference room at {}, after which they proceeded with the quarterly report.",
    "I remember walking into the hospital around {}; the receptionist looked surprised to see me that early.",
    "It was {}, and the air was already humid. We knew it was going to be a long day.",
    "By {}, the doctor had finished all his morning rounds and went to brief the family.",
    "We were finally discharged at {}. It had been a tough week of waiting and uncertainty.",
    "He usually goes jogging around {}, but today he stayed in because it was raining heavily.",
    "She had breakfast at {}, took a quick shower, and left for the appointment feeling better.",
    "Back then, they always rang the bell at {}, a practice the ward followed strictly.",
    "It wasn’t until {} that I realized she hadn’t shown up for her therapy session.",
    "Around {}, the second nurse came in and adjusted the IV. The patient didn’t even notice.",
    "I fell asleep sometime after {}, exhausted from the stress and noise of the ER.",
    "They gave him the injection at {}, then monitored his vitals for another 30 minutes.",
    "He skipped lunch and only ate around {} when the nurse reminded him twice.",
    "The worst part came at {}; I couldn’t bear to hear the results but had no choice.",
    "The ambulance arrived at {}, and within minutes, the entire street was blocked off.",
    "They told us to wait outside until {}, saying the procedure would take time to complete.",
    "At {}, her heart rate started to stabilize, and everyone in the room breathed easier.",
    "He returned from the break room at {}, just in time to hear the announcement.",
    "Before {}, we still had hope. But after that, the entire tone changed.",
    "I kept checking the time, and when it hit {}, I knew I had to leave.",
    "The symptoms became severe around {}, and we rushed to get a doctor immediately.",
    "She mentioned feeling dizzy since {}, but thought it was just lack of sleep.",
    "I started documenting the changes exactly at {}, just as I had been instructed.",
    "We finally got a callback at {}, hours after we had left the first message.",
    "He last took his meds at {}, which might explain the drop in pressure.",
    "The lights dimmed at {}, marking the start of the overnight quiet zone.",
    "At {}, they switched shifts, and a new nurse came to take over the monitoring.",
    "I remember the clock striking {} while we were still waiting in the hallway.",
    "She said the pain started around {}, and it gradually worsened from there.",
    "Just after {}, they rolled her into the operating room and closed the doors.",
    "I got a message at {}, telling me to come as soon as possible.",
    "By the time it was {}, the lab results had already been uploaded to the system.",
    "He called again around {}, sounding more confused than before.",
    "We didn’t hear anything until {}, and even then, it wasn’t very clear.",
    "Around {}, the fire alarm accidentally went off, sending everyone outside in their gowns.",
]


# 更多自然時間片語
time_phrases = [
    "this morning", "this evening", "tonight", "last night", "yesterday morning",
    "tomorrow afternoon", "Tuesday morning", "Wednesday night", "Friday evening",
    "in the early morning", "late at night", "middle of the night", "lunchtime",
    "bedtime", "dawn", "dusk", "midnight", "noon", "around 5-ish", "about midnight",
    "by lunchtime", "later today", "after dinner", "just before bed","five o'clock", "six o'clock", "ten o'clock", "three o'clock",
    "half past six", "quarter past eight", "quarter to nine",
    "ten past seven", "twenty to five", "about five o'clock",
    "just after six o'clock", "a little past nine"
]

# 時間辨識規則
def find_time_mentions(text):
    patterns = [
        r"\b\d{1,2}(:\d{2})?\s*(AM|PM|am|pm)?",
        r"\b(this|last|next|yesterday|tomorrow|another)\s+(morning|afternoon|evening|night)",
        r"\b(early|late|middle of the|second)\s+(night|morning|afternoon|evening)",
        r"\b(lunchtime|bedtime|noon|midnight|dawn|dusk)",
        r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(morning|night|afternoon|evening)",
        r"\b(?:around|by|about|after|before|just before)\s+\d{1,2}(?:-\w+|(?:\s*ish))?(?:\s+(this|last|next|yesterday|tomorrow))?\s*(morning|afternoon|evening|night)?",
        # ✅ 新增：口語式時間（five o'clock, quarter to six）   
        r"\b(?:five|six|seven|eight|nine|ten|eleven|twelve|one|two|three|four)\s+o'clock\b",
        r"\b(?:quarter|half|ten|twenty|five)\s+(?:past|to)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    ]
    results = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            results.append((float(m.start()), float(m.end()), m.group().strip()))
    return results

# 主資料生成函式
def generate_time_data(filename1, filename2, start_sid=24000, total=200):
    task1, task2 = [], []
    sid = start_sid
    used_sentences = set()

    while len(task1) < total:
        if random.random() < 0.5:
            hour = random.randint(0, 23)
            minute = random.choice([0, 15, 30, 45])
            time_str = f"{hour}:{minute:02d}"
            if random.random() < 0.5:
                time_str += " " + ("AM" if hour < 12 else "PM")
        else:
            time_str = random.choice(time_phrases)

        template = random.choice(time_templates_rich)
        sentence = template.format(time_str)

        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)

        task1.append(f"{sid}\t{sentence}")
        found = False
        for start, end, phrase in find_time_mentions(sentence):
            if time_str.split()[0].lower() in phrase.lower() or time_str.lower() in phrase.lower():
                task2.append(f"{sid}\tTIME\t{start:.3f}\t{end:.3f}\t{phrase}")
                found = True
                break

        if not found:
            match = re.search(re.escape(time_str.split()[0]), sentence, flags=re.IGNORECASE)
            if match:
                task2.append(f"{sid}\tTIME\t{float(match.start()):.3f}\t{float(match.end()):.3f}\t{time_str}")

        sid += 1

    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f"✅ 已產生 {len(task1)} 筆 TIME 樣本與 {len(task2)} 筆標註")

# ✅ 執行
generate_time_data("task1_time.txt", "task2_time.txt", start_sid=60000, total=200)
