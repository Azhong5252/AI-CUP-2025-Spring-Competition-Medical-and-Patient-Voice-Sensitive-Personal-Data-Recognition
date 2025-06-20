from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
import re
def run():
    # 裝置設置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型與 tokenizer 載入
    model_dir = "model/ner_model_doctor"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    id2label = model.config.id2label

    # 測試資料（Whisper 輸出文字）
    with open("ASR_code/text/Whisper_Validation.txt", encoding='utf-8') as f:
        lines = dict(line.strip().split('\t', 1) for line in f if line.strip())

    # 實體過濾規則
    def clean_entity_text(label, token_text):
        # 排除數字
        if any(char.isdigit() for char in token_text):
            return None

        # 排除像 "Dr.A-Z.A-Z" 這樣由單一字母構成的字符串
        # if re.match(r"^[A-Za-z\.-]+$", token_text):
        #     return None

        # 排除 "doctor" 或 "DOCTOR"
        if token_text.lower() == "doctor":
            return None

        # 如果是 "Dr" 且後面跟有名字或其他字符，則保留
        if label == "DOCTOR" and token_text.lower().startswith("dr "):
            # 檢查是否有名字或者有效的字母字符
            if len(token_text.split()) > 1 and re.match(r"[A-Za-z]+", token_text.split()[1]):
                return token_text.strip()  # 保留有效的醫生名稱
        
        return token_text.strip()

    # 推理主函數
    def predict_entities(text_id, text, confidence_threshold=0.8):
        encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
        offset_mapping = encoding.pop("offset_mapping")[0]
        word_ids = encoding.word_ids()
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

        result = []
        current_entity = []
        current_label = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            label = id2label[predictions[0][i].item()]
            confidence = confidences[0][i].item()
            start_char, end_char = offset_mapping[i].tolist()

            if confidence < confidence_threshold or label == "O":
                if current_entity:
                    full_text = text[current_entity[0][0]:current_entity[-1][1]]
                    cleaned = clean_entity_text(current_label, full_text)
                    if cleaned:
                        result.append((text_id, current_label, cleaned))
                    current_entity = []
                    current_label = None
                continue

            label_type = label[2:] if label.startswith("B-") else None
            if label.startswith("B-"):
                if current_entity:
                    full_text = text[current_entity[0][0]:current_entity[-1][1]]
                    cleaned = clean_entity_text(current_label, full_text)
                    if cleaned:
                        result.append((text_id, current_label, cleaned))
                current_entity = [(start_char, end_char)]
                current_label = label_type
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append((start_char, end_char))
            else:
                if current_entity:
                    full_text = text[current_entity[0][0]:current_entity[-1][1]]
                    cleaned = clean_entity_text(current_label, full_text)
                    if cleaned:
                        result.append((text_id, current_label, cleaned))
                current_entity = []
                current_label = None

        if current_entity:
            full_text = text[current_entity[0][0]:current_entity[-1][1]]
            cleaned = clean_entity_text(current_label, full_text)
            if cleaned:
                result.append((text_id, current_label, cleaned))

        return list(sorted(set(result), key=lambda x: (x[0], x[1])))

    def filter_doctors(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 去除行首和行尾的空白字符（包括換行符）
                line = line.strip()

                # 解析每一行，假設格式是 "ID DOCTOR NAME"
                parts = line.split('\t')
                if len(parts) != 3:
                    continue  # 跳過格式不正確的行
                
                # 確保只有 "DOCTOR" 標籤的行
                doc_id, label, name = parts
                if label != "DOCTOR":
                    continue

                # 如果名字開頭不是 "Dr " 或 "Dr."，則跳過這一行
                if not (name.startswith("Dr ") or name.startswith("Dr.") or name.startswith("Dr. ")):
                    continue  # 這會跳過所有不是以 "Dr" 開頭的行

                
                outfile.write(line + '\n')

        print("✅ 過濾完成，結果儲存於", output_file)

    # 儲存推論結果
    os.makedirs("validation", exist_ok=True)
    with open("validation/inference_doctor_output.txt", "w", encoding="utf-8") as out:
        for sid, text in lines.items():
            preds = predict_entities(sid, text)
            for filename, label, val in preds:
                print(f"{filename}\t{label}\t{val}")
                out.write(f"{filename}\t{label}\t{val}\n")

    print("✅ 推論完成，結果儲存於 validation/inference_doctor_output.txt")

    # #移除不是Dr開頭的
    filter_doctors("validation/inference_doctor_output.txt", "validation/doctor_output.txt")

    # 2. 將 validation/doctor_output.txt 複寫回 validation/inference_doctor_output.txt
    with open("validation/doctor_output.txt","r",encoding="utf-8") as f:
        doctor_output_content = f.read()
    with open("validation/inference_doctor_output.txt","w",encoding="utf-8") as f:
        f.write(doctor_output_content)
    # #移除doctor_output.txt
    os.remove("validation/doctor_output.txt")
if __name__ == "__main__":
    run()