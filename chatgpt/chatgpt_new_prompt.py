import re
import time
from openai import OpenAI

client = OpenAI(api_key="")

messages = []

ALL_SHI_TYPES = [
    "PATIENT", "DOCTOR", "USERNAME", "FAMILYNAME", "PERSONALNAME", "PROFESSION", 
    "ROOM", "DEPARTMENT", "HOSPITAL", "ORGANIZATION", "STREET", "CITY", 
    "DISTRICT", "COUNTY", "STATE", "COUNTRY", "ZIP", 
    "AGE", "DATE", "TIME", "DURATION", "SET", "PHONE", "FAX", "EMAIL", 
    "URL", "IPADDRESS", "SOCIAL_SECURITY_NUMBER", "MEDICAL_RECORD_NUMBER", 
    "HEALTH_PLAN_NUMBER", "ACCOUNT_NUMBER", "LICENSE_NUMBER", "VEHICLE_ID","BIOMETRIC_ID", "ID_NUMBER","LOCATION-OTHER"
]

def gpt(message):
    messages.clear()
    prompt = f"""
    From the following text, identify and label all entities that belong to the following categories: {", ".join(ALL_SHI_TYPES)}.
    Please follow the definitions of each type. Return each matched entity in the format:
    filename SHI_TYPE text

    Definitions of selected SHI types:
    FAMILYNAME: The patient's immediate family members, spouse, and children.
    CARETAKER: A person who assists in daily care tasks but is not a family member or a doctor, such as a nurse, assistant, or home aide.

    Examples of recognized entities:
    PATIENT Mrs. Smith
    PATIENT Ramona
    PATIENT June
    PATIENT Ken Moll
    DOCTOR Dr. Alex Lamb
    DOCTOR Dr. Yale
    DOCTOR Dr. A. Duncan-Tail
    DOCTOR Dr. A. Royce-Coe
    DOCTOR Dr. Alex Lamb
    DOCTOR Dr. Goldstein
    USERNAME Smi123
    FAMILYNAME Ivan's dad
    FAMILYNAME Ivan's mon
    HOSPITAL St. George Hospital
    HOSPITAL Prince of Wales Hospital
    DEPARTMENT 8.MEDIC/SURGERY WARD-POWP
    ROOM Room 12
    ROOM OBG floor
    ROOM Room3
    STREET 12 Kings Avenue
    STREET 345 Beach Street
    CITY Randwick
    CITY Bondi
    CITY Los Angeles
    DISTRICT Sanmin District
    COUNTY Cheshire
    COUNTY Los Angeles County
    STATE NSW
    STATE ACT
    STATE California
    ZIP 2344
    ZIP 2567
    COUNTRY Australia
    COUNTRY USA
    COUNTRY South Africa
    AGE 52
    AGE 20
    AGE 32
    AGE 40
    AGE 82
    AGE 60
    DATE July 4, 2020
    DATE 12, 2063
    DATE September 11
    DATE now
    DATE August
    DATE tomorrow
    DATE today
    DATE Ash Wednesday
    DATE Christmas
    DATE Last week
    TIME 1:00 AM
    TIME five o'clock
    TIME Midnight
    TIME this morning
    TIME last night
    DURATION 15 minutes
    DURATION three months
    DURATION two weeks
    DURATION coming weeks
    DURATION months
    DURATION over time
    DURATION three hours
    DURATION two hours
    DURATION two years
    SET 30 a week
    SET twice
    SET couple of times
    SET five times per month
    MEDICAL_RECORD_NUMBER 022213.PWP
    MEDICAL_RECORD_NUMBER 55555774.RAN
    VEHICLE_NUMBER CB 33 GO
    VEHICLE_NUMBER LCS 23
    VEHICLE_NUMBER CC 12 MS
    ID_NUMBER 12R423044B
    ID_NUMBER 12H08861
    ID_NUMBER B1
    ID_NUMBER S123456789
    PHONE 02-2222-3455
    PHONE 8282 7154
    FAX 02-2222-3455
    FAX 8123 9876 
    EMAIL abc@gmail.com
    URL www.xyz.com
    IPADDRESS 192.1.1.1
    Do not translate the text content.

    examples:
    input:23	Yeah, I imagine it would — sorry, go ahead. So it's supposed to work immediately, right? Yep. So we'll see if I'm productive tomorrow. I hope I'm productive today. I've actually been trying to plan. If I do the titles today, then I can do my laundry tomorrow. Right. I probably could bring my computer and do titles while I'm doing my laundry. If I was — but I won't do that.
    output:
    23	DATE tomorrow
    23	DATE today

    Now, apply this format to the following text:
    {message}
    """
    messages.append({"role": "user", "content": message})
    messages.append({"role": "system", "content": prompt})

    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100,
        temperature=0.2
    )

    reply = chat_completion.choices[0].message.content

    cleaned_reply = reply.replace("```", "").strip()

    messages.append({"role": "assistant", "content": reply})
    return cleaned_reply

def append_to_gpt_file(text, filename="validation/GPT.txt", original_line=None):
    valid_lines = []
    for line in text.strip().split("\n"):
        match = re.match(r"^(\d+)\t([A-Z_]+)\s+(.+)", line.strip())
        if match:
            utt_id, ent_type, ent_text = match.groups()
            if ent_type not in ALL_SHI_TYPES:
                continue
            if ent_text.strip().lower() == ent_type.strip().lower():
                continue
            if original_line and ent_text in original_line:
                valid_lines.append(line.strip())
    if valid_lines:
        with open(filename, "a", encoding="utf-8") as f:
            f.write("\n".join(valid_lines) + "\n")

with open("ASR_code/text/Whisper_Validation.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    if "\t" not in line:
        continue  

    utt_id, text = line.strip().split("\t", 1)
    full_text = f"{utt_id}\t{text}"

    try:
        print(f"處理段落 {utt_id}...")
        result = gpt(full_text)
        print(result)
        append_to_gpt_file(result, original_line=text)
        time.sleep(10)
    except Exception as e:
        print(f"段落 {utt_id} 發生錯誤: {e}")