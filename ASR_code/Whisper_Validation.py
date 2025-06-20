import whisper
import os
import time
import json
import re
from tqdm import tqdm

#可以使用的模型tiny/base/small/medium/large/large-v2/large-v3
model = whisper.load_model("large-v2", device="cuda")

Whisper_text = []
audio_dir = "ASR_code/audio"
output_text_path = "ASR_code/text/Whisper_Validation.txt"
output_time_path = "ASR_code/text/Whisper_Validation_Timestamps.json"

all_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
sorted_files = sorted(all_files, key=lambda x: int(os.path.splitext(x)[0]))

timestamps_data = {}

def clean_word(word):
    word = re.sub(r"[’']s\b", "s", word)

    word = re.sub(r"(\d)([a-zA-Z])", r"\1.\2", word)

    word = re.sub(r"[^\w.\-]", "", word)

    return word.strip()

for filename in tqdm(sorted_files, desc="轉換音檔"):
    audio_path = os.path.join(audio_dir, filename)
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        condition_on_previous_text = False,
        temperature=0.0,        
        beam_size=4,            
        best_of=5,               
        fp16=False             
    )

    file_number = os.path.splitext(filename)[0]

    Whisper_text.append(f"{file_number}\t{result['text'].strip()}")

    print(f"推理結果（{file_number}）：{result['text']}")

    word_times = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            word_clean = clean_word(word["word"])
            word_times.append({
                "word": word["word"],
                # "word": word_clean,
                "start": round(word["start"], 3),
                "end": round(word["end"], 3)
            })

    timestamps_data[file_number] = word_times
    # time.sleep(2)

os.makedirs("text", exist_ok=True)

with open(output_text_path, "w", encoding="utf-8") as f:
    for line in Whisper_text:
        print(line)
        f.write(line + "\n")

with open(output_time_path, "w", encoding="utf-8") as f:
    json.dump(timestamps_data, f, ensure_ascii=False, indent=2)

print("✅ Whisper 完成，結果儲存於：")
print(f"- {output_text_path}")
print(f"- {output_time_path}")