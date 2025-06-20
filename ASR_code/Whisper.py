import os
import json
import re
from faster_whisper import WhisperModel
from tqdm import tqdm
import gc
import torch

def run():
    def normalize_text(text):
        text = re.sub(r"\s+", " ", text)  
        return text.strip()


    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    audio_folder = "ASR_code/audio"
    txt_output_path = "ASR_code/text/Whisper_Validation.txt"
    normalized_txt_output_path = "ASR_code/text/Whisper_Validation_Normalized.txt"
    json_output_path = "ASR_code/text/Whisper_Validation_Timestamps.json"

    transcript_lines = []
    normalized_lines = []
    timestamp_data = {}

    audio_files = sorted(
        [f for f in os.listdir(audio_folder) if f.lower().endswith((".wav", ".mp3", ".m4a"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    for filename in tqdm(audio_files, desc="🔊 Transcribing audio files"):
        file_path = os.path.join(audio_folder, filename)
        file_id = os.path.splitext(filename)[0]

        segments, _ = model.transcribe(
            file_path,
            word_timestamps=True,
            vad_filter=False, 
            beam_size=20,      
            temperature=[0.0, 0.2, 0.4]
        )

        full_text = ""
        words = []

        for segment in segments:
            full_text += segment.text.strip() + " "
            if segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word.strip(),
                        "start": round(word.start, 2),
                        "end": round(word.end, 2)
                    })

        normalized_text = normalize_text(full_text)

        transcript_lines.append(f"{file_id}\t{full_text.strip()}")
        normalized_lines.append(f"{file_id}\t{normalized_text}")
        tqdm.write(f"{file_id}\t{normalized_text}")
        timestamp_data[file_id] = words

    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines))

    with open(normalized_txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(normalized_lines))

    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(normalized_lines))

    if os.path.exists(normalized_txt_output_path):
        os.remove(normalized_txt_output_path)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(timestamp_data, f, indent=2)

    print("✅ 轉換完成，已應用優化設定與保留標點。")
if __name__ == "__main__":
    run()