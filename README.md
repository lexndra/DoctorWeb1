# DoctorWeb1
Project by students NSU group 25948 Institute of Intelligent Robotics. A neural network for recognizing artificial intelligence in documents of various types.

## Модель сегментации ИИ-текста (TNC-DBGC*)

Архитектура: `Transformer (mDeBERTa-v3) + BiGRU + CRF` для пословного выявления границ перехода между текстом человека и нейросетей.

### Результаты валидации:
- **Accuracy:** 90.6%
- **F1-Score:** 90.6%
- **Cohen's Kappa:** 0.8120
- **Boundary MAE (ошибка границы):** 17.92 слов
- **F1@3 (попадание в стыки):** 62.7%

*Веса модели (`tnc_dbgc_coat_best.pt`, 1.1 ГБ) сохранены на Google Диске.*  
[Скачать веса модели tnc_dbgc_coat_best.pt](https://drive.google.com/file/d/1sfdBo-n8XDWCsi6bDoREvZMkHAf_ywbe/view?usp=sharing)
