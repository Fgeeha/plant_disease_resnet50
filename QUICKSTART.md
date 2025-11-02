# 🚀 Быстрый старт

## За 5 минут до обучения модели!

### 1️⃣ Клонируйте проект и перейдите в директорию

```bash
git clone <your-repo>
cd plant-disease-detector
```

### 2️⃣ Запустите скрипт установки

```bash
chmod +x setup.sh
./setup.sh
```

Скрипт автоматически:
- ✅ Проверит все зависимости
- ✅ Установит NVIDIA Docker (если нужно)
- ✅ Создаст структуру директорий
- ✅ Соберет Docker образ
- ✅ Проверит GPU

### 3️⃣ Подготовьте данные

**Вариант A: Скачать PlantVillage датасет**

```bash
# Установите Kaggle API
pip install kaggle

# Добавьте credentials в ~/.kaggle/kaggle.json
# Получите их на https://www.kaggle.com/settings

# Скачайте датасет
make download-plantvillage
```

**Вариант B: Использовать свои данные**

Организуйте в структуру:
```
data/
├── train/
│   ├── healthy/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── rust/
│   └── blight/
└── val/
    ├── healthy/
    ├── rust/
    └── blight/
```

### 4️⃣ (Опционально) Настройте конфигурацию

Отредактируйте `config.yaml` под ваши нужды:

```yaml
training:
  batch_size: 32        # Уменьшите если мало памяти
  num_epochs: 50
  learning_rate: 0.001

model:
  name: 'resnet50'     # или 'resnet101'
  num_classes: 38      # Обновится автоматически
```

### 5️⃣ Запустите обучение!

```bash
make train
```

### 6️⃣ Мониторьте процесс в TensorBoard

В отдельном терминале:

```bash
make tensorboard
```

Откройте в браузере: **http://localhost:6006**

---

## 📊 После обучения

### Предсказание

```bash
# Создайте файл test_inference.py
python3 -c "
from src.inference import PlantDiseasePredictor

predictor = PlantDiseasePredictor('models/plant_disease_resnet50.pth')
result = predictor.predict('path/to/image.jpg')
print(f\"Класс: {result['top_prediction']['class']}\")
print(f\"Уверенность: {result['top_prediction']['percentage']}\")
"
```

### Оценка модели

```bash
make evaluate
```

---

## 🆘 Частые проблемы

### GPU не обнаружен

```bash
# Проверьте драйвер
nvidia-smi

# Проверьте Docker
make check-gpu
```

### Ошибка памяти

Уменьшите batch_size в `config.yaml`:
```yaml
training:
  batch_size: 16  # Вместо 32
```

### Медленное обучение

Включите mixed precision:
```yaml
gpu:
  mixed_precision: true
```

---

## 📚 Полезные команды

```bash
make help              # Все доступные команды
make shell             # Открыть bash в контейнере
make jupyter          # Запустить Jupyter Lab
make logs             # Показать логи
make stop             # Остановить все контейнеры
```

---

## 🎯 Следующие шаги

1. Экспериментируйте с гиперпараметрами в `config.yaml`
2. Попробуйте разные модели (resnet101, efficientnet)
3. Добавьте больше аугментаций
4. Используйте transfer learning с fine-tuning

**Успехов! 🌿**