import random
import re
from collections import defaultdict

# 自然 + 正式 + 口語 混合句型 (Doctor專用)
doctor_templates_mixed = [
    "Dr. {} will be overseeing the patient's recovery process.",
    "The surgery was performed successfully by Dr. {}.",
    "Medical consultation was scheduled with Dr. {} next Tuesday.",
    "Dr. {} is known for expertise in cardiology.",
    "The patient reported improvement after seeing Dr. {}.",
    "Dr. {} recommended a new course of antibiotics.",
    "Follow-up appointments with Dr. {} have been arranged.",
    "The lab results were discussed with Dr. {}.",
    "Dr. {} examined the X-ray scans carefully.",
    "A letter of referral was issued by Dr. {}.",
    "Upon discharge, Dr. {} provided specific care instructions.",
    "The second opinion was sought from Dr. {}.",
    "Dr. {} specializes in pediatric care.",
    "A telehealth session was conducted by Dr. {}.",
    "According to Dr. {}, the prognosis is favorable.",
    "Dr. {} emphasized the importance of medication adherence.",
    "The operation notes were written by Dr. {}.",
    "The emergency department contacted Dr. {} for further guidance.",
    "Dr. {} is currently leading the research project.",
    "Patient history was updated by Dr. {} during rounds.",
    "Dr. {} is an expert in infectious diseases and patient management.",
    "The medical team consulted Dr. {} on the patient's condition.",
    "Dr. {} conducted the initial consultation with the patient.",
    "Dr. {} is known for providing excellent care in emergency situations.",
    "The patient’s treatment was adjusted based on Dr. {}’s recommendations.",
    "Dr. {} recommended a new medication plan to optimize recovery.",
    "Dr. {} helped the patient manage chronic conditions effectively.",
    "During the clinical trial, Dr. {} monitored patient responses closely.",
    "Dr. {} discussed treatment options with the patient’s family.",
    "A comprehensive health check was conducted by Dr. {} for the patient’s annual exam.",
    "Dr. {} performed a routine check-up and prescribed preventive measures.",
    "Dr. {} was instrumental in diagnosing the patient’s rare condition.",
    "Dr. {} helped manage the patient’s rehabilitation after surgery.",
    "Dr. {} recommended physical therapy following the patient's injury.",
    "Dr. {} has provided expertise in managing complex neurological cases.",
    "Dr. {} performed the medical evaluation during the patient’s admission.",
    "Dr. {} authored several research papers on mental health.",
    "The patient's response to treatment improved under the care of Dr. {}.",
    "Dr. {} reviewed the patient’s medical history before making a diagnosis.",
    "As the chief surgeon, Dr. {} led the successful operation.",
    "Dr. {} is a renowned expert in pediatric cardiology."
]

# Doctor名字資料
doctor_names = [
    "John Smith", "Emily Johnson", "Michael Williams", "Sarah Brown", "David Jones",
    "Laura Garcia", "James Miller", "Olivia Davis", "Daniel Martinez", "Sophia Rodriguez",
    "Matthew Hernandez", "Emma Lopez", "Christopher Gonzalez", "Ava Wilson",
    "Anthony Anderson", "Mia Thomas", "Joshua Taylor", "Abigail Moore",
    "Andrew Jackson", "Isabella Martin"
]

# 找到 Dr.名字 的位置
def find_doctor_mentions(text):
    pattern = r"Dr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"  # Ex: Dr. John Smith
    results = []
    for m in re.finditer(pattern, text):
        mention = m.group()
        results.append((float(m.start()), float(m.end()), mention.strip()))
    return results

# 生成 DOCTOR 資料
def generate_doctor_data(filename1, filename2, start_sid=70000, total=500):
    task1, task2 = [], []
    sid = start_sid
    used_sentences = set()

    while len(task1) < total:
        name = random.choice(doctor_names)
        sentence = random.choice(doctor_templates_mixed).format(name)

        # 避免重複句子
        if sentence in used_sentences:
            continue
        used_sentences.add(sentence)

        task1.append(f"{sid}\t{sentence}")
        found = False
        # 這裡只檢查一次 Dr.名字位置
        for start, end, phrase in find_doctor_mentions(sentence):
            if name in phrase:
                task2.append(f"{sid}\tDOCTOR\t{start:.1f}\t{end:.1f}\t{phrase}")
                found = True

        # 如果沒有找到Dr.名字，則打印警告
        if not found:
            print(f"⚠️ Warning: 未標註到 Dr.名字 in句子: {sentence}")

        sid += 1

    # 寫入文件
    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f"✅ DOCTOR 資料已產生，共 {len(task1)} 筆句子與 {len(task2)} 筆標註")

# 使用範例
if __name__ == "__main__":
    generate_doctor_data("task1_doctor.txt", "task2_doctor.txt")