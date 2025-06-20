import random
from collections import defaultdict

shi_type_examples = {
    "DATE": [
        "now", "tomorrow", "today", "this week", "Last week", 
        "September 11", "March 25, 2021", "June 1, 2022", "October 15, 2021", 
        "January 1, 2022", "April 23, 2025", "May 10, 2024", "September 15", 
        "February 14", "August 22", "November 30",
        "January 1", "January 15", "January 30", 
        "February 1", "February 14", "February 28",
        "March 1", "March 15", "March 30", 
        "April 1", "April 10", "April 23", 
        "May 1", "May 10", "May 25",
        "June 1", "June 5", "June 20",
        "July 1", "July 4", "July 10", 
        "August 1", "August 15", "August 22", 
        "September 1", "September 5", "September 15", 
        "October 1", "October 15", "October 30", 
        "November 1", "November 11", "November 30", 
        "December 1", "December 10", "December 25"
    ]
}
def make_long_sentence(shi_type, value):
    if shi_type == "DATE":
        templates = [
            f"The scheduled appointment on {value} marks a key checkpoint in the treatment cycle.",
            f"Patient progress will be reviewed thoroughly during the meeting on {value}.",
            f"The follow-up session has been confirmed for {value}, and reminders have been sent to all involved parties.",
            f"Tests conducted on {value} revealed important updates regarding the patient's condition.",
            f"The surgery is tentatively scheduled for {value}, pending final approval from the surgical team.",
            f"Team members should submit their reports before {value} to prepare for the upcoming evaluation.",
            f"On {value}, a comprehensive diagnostic session will be held to reassess treatment goals.",
            f"{value} has been selected as the starting date for the next treatment phase.",
            f"All prescriptions should be renewed by {value} to avoid any disruption in care.",
            f"The care plan will be updated after the consultation on {value}.",
            f"By {value}, we expect measurable improvements in motor function based on current therapy.",
            f"Data collection will end on {value}, after which analysis will begin.",
            f"{value} is the deadline for submitting insurance documentation.",
            f"The official discharge process begins on {value}, assuming all conditions are stable.",
            f"{value} is critical for completing lab work required for the next procedure.",
            f"Post-operative assessments are due on {value}.",
            f"{value} was initially missed but has now been rescheduled with priority.",
            f"All records from before {value} will be archived under the new protocol.",
            f"Medication adjustments will be evaluated on {value} depending on blood test results.",
            f"Please arrive early on {value} for pre-op clearance and final checks."
        ]
        return random.choice(templates)
    return f"[{shi_type}] {value}"

def normalize_entity_for_span(t, val):
    return val

def generate_date_data(filename1, filename2, start_sid=10000, total=200):
    from collections import defaultdict
    task1, task2 = [], []
    sid = start_sid
    counts = defaultdict(int)
    shi_type = "DATE"
    values = shi_type_examples[shi_type]

    while sid < start_sid + total and counts[shi_type] < total:
        val = random.choice(values)
        norm_val = normalize_entity_for_span(shi_type, val)
        if not norm_val:
            continue

        sentence = make_long_sentence(shi_type, val)
        if norm_val not in sentence:
            sentence += f" ({norm_val})"

        start = sentence.index(norm_val)
        end = start + len(norm_val)

        task1.append(f"{sid}\t{sentence}")
        task2.append(f"{sid}\t{shi_type}\t{start:.1f}\t{end:.1f}\t{norm_val}")
        counts[shi_type] += 1
        sid += 1

    with open(filename1, "w", encoding="utf-8") as f1, open(filename2, "w", encoding="utf-8") as f2:
        f1.write("\n".join(task1))
        f2.write("\n".join(task2))

    print(f"DATE 資料已產生，共 {len(task1)} 筆句子與 {len(task2)} 筆標註")

generate_date_data("task1_date.txt", "task2_date.txt")
