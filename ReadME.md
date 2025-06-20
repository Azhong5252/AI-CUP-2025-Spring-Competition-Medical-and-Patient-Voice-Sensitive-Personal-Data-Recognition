# AI CUP 2025 春季賽 醫病語音敏感個人資料辨識競賽賽

## 環境準備

在執行專案前，請先建立虛擬環境：

```bash
conda create -n env_name python==3.10.0 --yes
conda activate env_name
# 若需移除環境
conda remove -n env_name --all --yes
```

## 安裝必要套件

請依序執行以下指令來安裝依賴：

```bash
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install tokenizers==0.21.1
pip install git+https://github.com/openai/whisper.git
pip install faster-whisper==0.9.0
pip install --upgrade huggingface-hub
```

安裝完成後，請執行以下指令檢查套件版本：

```bash
pip list
```

需確認以下版本：

- `huggingface-hub`: **0.33.0**
- `tokenizers`: **0.21.1**

若版本不符，請使用以下方式重新安裝：

```bash
pip install --upgrade huggingface-hub
pip install tokenizers==0.21.1
```

## 第一階段：醫病語音紀錄辨識（ASR）

1. 將音檔放入資料夾：  
   `Train_sigle_model_deberta/ASR_code/audio/`

2. 有兩種方法可以執行 ASR：

### 方法一：OpenAI Whisper

請確認當前路徑為：

```
C:\Users\user\Desktop\AI CUP資源\Train_sigle_model_deberta>
```

然後執行：

```bash
python .\ASR_code\Whisper_Validation.py
```

執行完成後，將會產生以下兩個檔案於 `ASR_code/text/`：

- `Whisper_Validation_Timestamps.json`
- `Whisper_Validation.txt` ← 這就是 Task 1 的輸出結果

---

### 方法二：Faster Whisper

請確認當前路徑為：

```
C:\Users\user\Desktop\AI CUP資源\Train_sigle_model_deberta>
```

然後執行：

```bash
python .\ASR_code\Whisper.py
```

執行完成後，將會產生以下兩個檔案於 `ASR_code/text/`：

- `Whisper_Validation_Timestamps.json`
- `Whisper_Validation.txt` ← 這就是 Task 1 的輸出結果

---

## 第二階段：醫病語音隱私個資辨識

同樣有兩種方法可以選擇：

### 方法一：使用 OpenAI API 呼叫

確認當前路徑為：

```
C:\Users\user\Desktop\AI CUP資源\Train_sigle_model_deberta>
```

然後執行：

```bash
python .\chatgpt\chatgpt_new_prompt.py
```

推理結果將會儲存在：

```
validation/GPT.txt
```

這個 `GPT.txt` 檔案即為 Task 2 的最終結果。

### 方法二：使用 DeBERTa-v3-base 模型進行 Fine-tuning

確認當前路徑為：

```
C:\Users\user\Desktop\AI CUP資源\Train_sigle_model_deberta>
```

然後依序執行以下訓練指令：

```bash
python .\train_code\train_age.py
python .\train_code\train_date.py
python .\train_code\train_doctor.py
python .\train_code\train_duration.py
python .\train_code\train_id_number.py
python .\train_code\train_location.py
python .\train_code\train_medical_record.py
python .\train_code\train_name.py
python .\train_code\train_profession.py
python .\train_code\train_set.py
python .\train_code\train_time.py
python .\train_code\train_zip.py
```

訓練完成後，開始進行推理：

```bash
python main.py
```

推理結果將會儲存在：

```
validation/inference_output.txt
```

這個 `inference_output.txt` 檔案即為 Task 2 的最終結果。
##
第一階段選擇faster-whisper，第二階段選擇deberta-v3-base分數如下
- `0.16/0.45`

第一階段選擇faster-whisper，第二階段選擇Openai-API呼叫分數如下
- `0.16/0.54`