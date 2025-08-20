# Real-or-Spam SMS Detection 📱🛡️

Türkçe SMS mesajları için geliştirilmiş yapay zeka tabanlı spam tespit sistemi. Bu proje, BERT tabanlı derin öğrenme modeli kullanarak SMS mesajlarını "ham" (normal) veya "spam" (istenmeyen) olarak sınıflandırır.

## 🎯 Proje Özeti

Bu proje, Türkçe SMS mesajlarını otomatik olarak sınıflandırmak için geliştirilmiş bir makine öğrenmesi uygulamasıdır. `dbmdz/bert-base-turkish-uncased` modeli kullanılarak fine-tuning yapılmış ve yüksek doğruluk oranları elde edilmiştir.

## 🚀 Özellikler

- **Türkçe Dil Desteği**: Türkçe metinler için optimize edilmiş BERT modeli
- **Yüksek Doğruluk**: %98+ doğruluk oranı
- **Gerçek Zamanlı Tahmin**: Yeni mesajlar için anında sınıflandırma
- **Görselleştirme**: Eğitim sürecinin detaylı analizi ve grafikleri
- **Model Kaydetme**: En iyi performans gösteren modelin otomatik kaydedilmesi

## 📊 Model Performansı

### Test Sonuçları (En İyi Model - Epoch 1)

| Metrik | Ham | Spam | Genel |
|--------|-----|------|-------|
| **Precision** | 0.98 | 0.97 | 0.98 |
| **Recall** | 0.97 | 0.98 | 0.98 |
| **F1-Score** | 0.98 | 0.98 | 0.98 |
| **Support** | 153 | 150 | 303 |

### Validation Sonuçları

| Metrik | Ham | Spam | Genel |
|--------|-----|------|-------|
| **Precision** | 0.99 | 0.99 | 0.99 |
| **Recall** | 0.99 | 0.99 | 0.99 |
| **F1-Score** | 0.99 | 0.99 | 0.99 |
| **Support** | 152 | 151 | 303 |

## 📈 Eğitim Grafikleri

### Eğitim ve Doğrulama Sonuçları
![Training Results](https://raw.githubusercontent.com/ByErenOzer/Real-or-Spam-SMS-Detection/main/training_results.png)

### Test Sınıflandırma Raporu
![Test Classification Report](https://raw.githubusercontent.com/ByErenOzer/Real-or-Spam-SMS-Detection/main/test_classification_report.png)

### Validation Sınıflandırma Raporu
![Validation Classification Report](https://raw.githubusercontent.com/ByErenOzer/Real-or-Spam-SMS-Detection/main/validation_classification_report.png)

## 🛠️ Kurulum

### Gereksinimler

```bash
pip install pandas numpy scikit-learn torch transformers tqdm seaborn matplotlib openpyxl
```

### Gerekli Kütüphaneler

- **pandas**: Veri manipülasyonu
- **numpy**: Sayısal hesaplamalar
- **scikit-learn**: Makine öğrenmesi metrikleri
- **torch**: PyTorch derin öğrenme framework'ü
- **transformers**: Hugging Face transformers kütüphanesi
- **tqdm**: İlerleme çubuğu
- **seaborn & matplotlib**: Veri görselleştirme
- **openpyxl**: Excel dosyası okuma

## 📁 Proje Yapısı

```
Real-or-Spam-SMS-Detection/
├── sms_spam_detection.py          # Ana eğitim scripti
├── karistirilmis_sms_dataset.xlsx  # Eğitim veri seti
├── sms_spam_detection/             # Model çıktıları
│   ├── best_sms_model/            # En iyi model
│   ├── best_sms_tokenizer/        # Tokenizer
│   ├── training_history.json      # Eğitim geçmişi
│   └── training_results.png       # Sonuç grafikleri
└── README.md                      # Bu dosya
```

## 🚀 Kullanım

### 1. Modeli Eğitme

```bash
python sms_spam_detection.py
```

### 2. Örnek Kullanım

Kod, eğitim sonrasında otomatik olarak 5 örnek mesaj ile test yapar:

```python
test_messages = [
    "Tebrikler! 1000 TL kazandınız! Hemen bu linke tıklayın: bit.ly/kazandin",
    "Merhaba, yarın saat 3'te toplantımız var. Unutma!",
    "ACIL! Hesabınız bloke edildi. Şifrenizi güncellemek için: fake-bank.com",
    "Anne, eve geç geleceğim. Akşam yemeğini bekleme.",
    "ÜCRETSİZ iPhone kazanmak için bu mesajı 10 kişiye gönder!"
]
```

### 3. Yeni Mesaj Tahmin Etme

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Modeli yükle
tokenizer = AutoTokenizer.from_pretrained('./sms_spam_detection/best_sms_tokenizer')
model = AutoModelForSequenceClassification.from_pretrained('./sms_spam_detection/best_sms_model')

# Mesajı tahmin et
message = "Yeni mesajınız var"
encoding = tokenizer.encode_plus(
    message.lower(),
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

with torch.no_grad():
    outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1).max().item()

label_names = ['HAM (Normal)', 'SPAM (İstenmeyen)']
print(f"Tahmin: {label_names[prediction]}")
print(f"Güven Skoru: {confidence:.4f}")
```

## 🔧 Model Detayları

### Kullanılan Model
- **Base Model**: `dbmdz/bert-base-turkish-uncased`
- **Task**: Binary Classification (Ham vs Spam)
- **Max Length**: 128 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 2
- **Optimizer**: AdamW

### Veri Ön İşleme
- Küçük harfe çevirme
- Fazla boşlukları temizleme
- Boş mesajları filtreleme
- Label encoding (ham: 0, spam: 1)

### Veri Bölünmesi
- **Eğitim**: %70
- **Doğrulama**: %15
- **Test**: %15

## 📊 Veri Seti

Proje, karışık Türkçe SMS veri seti kullanmaktadır:
- **Toplam Mesaj**: ~2000+ SMS
- **Ham Mesajlar**: ~1000+
- **Spam Mesajlar**: ~1000+
- **Dil**: Türkçe
- **Format**: Excel (.xlsx)

## 🎯 Sonuçlar

- **Test Doğruluğu**: %98
- **F1-Score**: 0.98
- **Precision**: 0.98
- **Recall**: 0.98

Model, hem ham hem de spam mesajları için yüksek performans göstermektedir.

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Geliştirici

**Eren Özer**
- GitHub: [@ByErenOzer](https://github.com/ByErenOzer)
- LinkedIn: [Eren Özer](https://linkedin.com/in/ereno)

## 🙏 Teşekkürler

- Hugging Face transformers kütüphanesi
- dbmdz Türkçe BERT modeli
- PyTorch ekibi
- Scikit-learn geliştiricileri

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!