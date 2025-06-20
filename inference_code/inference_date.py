from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification, pipeline
import re
import torch
def run():
    # 裝置設置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型與 tokenizer 讀取
    model_dir = "model/ner_model_date"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
    model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)

    # 使用 simple 聚合策略，自動合併連續實體詞
    ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

    # 資料路徑
    input_path = "ASR_code/text/Whisper_Validation.txt"
    output_path = "validation/inference_date_output.txt"

    def is_valid_date(text):
        text = text.strip().lower()
        
        # ⚠️ 排除只有 "this" 的情況
        if text == "this":
            return False
        
        date_keywords = ["today", "tomorrow", "yesterday", "now", "this", "last", "next"]
        months = r"(january|february|march|april|may|june|july|august|september|october|november|december)"

        if any(k in text for k in date_keywords):
            return True
        if re.search(rf"{months} \d{{1,2}}(, \d{{4}})?", text):
            return True
        if re.fullmatch(r"\d{1,2}, \d{4}", text):
            return True
        return False


    # 讀取資料
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []

    # 處理每行
    for line in lines:
        try:
            sid, sentence = line.split('\t', 1)
        except ValueError:
            continue

        ner_results = ner_pipeline(sentence)

        for r in ner_results:
            if r["entity_group"] == "DATE":
                entity_text = r["word"].replace("▁", " ").strip()

                # 濾除條件：
                if (
                    len(entity_text) <= 2 or
                    re.fullmatch(r"[\d\s,\.]+", entity_text) or
                    not is_valid_date(entity_text)
                ):
                    continue

                results.append(f"{sid}\tDATE\t{entity_text}")

    # 寫入結果
    with open(output_path, "w", encoding="utf-8") as f:
        for line in results:
            print(line)
            f.write(line + "\n")

    print(f"✅ 推理完成，結果已儲存至 {output_path}")
