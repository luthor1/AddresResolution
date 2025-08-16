from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Model ve tokenizer'ı kendi eğittiğin modelin kaydedildiği klasörden yükle
model_name_or_path = "model/fineModel"  # Örn: "./my_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Örnek cümle
sentence = ["Cumhuriye Mah. Hükümet Cad. Sivriler İşhanı No:3 Fethiye/Muğla Foto Kandiye Muğla / Fethiye",
            "Akarca Mah. Adnan Menderes Cad. 864.Sok. No:15 D.1 K.2 .",
            "Dedebaşı mahallesi 6100 sokak no 10 Kat 7 daire 25 ",
            ]
result = classifier(sentence)

print(result)

