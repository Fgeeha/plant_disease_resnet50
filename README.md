# 🌿 Plant Disease Detection with NVIDIA GPU

Проект для обучения модели классификации заболеваний растений с использованием Docker и NVIDIA GPU.

## 📋 Содержание

- [Требования](#требования)
- [Установка](#установка)
- [Структура проекта](#структура-проекта)
- [Быстрый старт](#быстрый-старт)
- [Использование Makefile](#использование-makefile)
- [Конфигурация](#конфигурация)
- [Обучение модели](#обучение-модели)
- [Инференс](#инференс)

## 🔧 Требования

### Система
- Ubuntu 18.04+ / Windows 10+ с WSL2
- NVIDIA GPU (с поддержкой CUDA 11.8+)
- Docker 20.10+
- NVIDIA Docker Runtime
- Make

### Проверка GPU
```bash
nvidia-smi
```

## 🚀 Установка

### 1. Установка NVIDIA Docker

```bash
# Добавление репозитория
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Установка
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Перезапуск Docker
sudo systemctl restart docker
```

### 2. Клонирование проекта

```bash
git clone <your-repo>
cd plant-disease-detector
```

### 3. Начальная настройка

```bash
make setup
```

## 📁 Структура проекта

```
plant-disease-detector/
├── Dockerfile              # Docker образ с PyTorch и CUDA
├── docker-compose.yml      # Docker Compose конфигурация
├── Makefile               # Команды для управления проектом
├── requirements.txt       # Python зависимости
├── config.yaml           # Конфигурация обучения
├── src/
│   ├── train.py          # Скрипт обучения
│   ├── inference.py      # Скрипт инференса
│   └── evaluate.py       # Скрипт оценки модели
├── data/
│   ├── train/           # Тренировочные данные
│   ├── val/             # Валидационные данные
│   └── test/            # Тестовые данные
├── models/              # Сохраненные модели
├── results/             # Результаты обучения
│   └── logs/           # TensorBoard логи
└── notebooks/          # Jupyter notebooks
```

## ⚡ Быстрый старт

### 1. Построение Docker образа

```bash
make build
```

### 2. Подготовка данных

Организуйте данные в следующую структуру:
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

**Или скачайте PlantVillage датасет:**
```bash
make download-plantvillage
```

### 3. Проверка GPU

```bash
make check-gpu
```

Вывод должен показать:
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX 3080
```

### 4. Запуск обучения

```bash
make train
```

### 5. Мониторинг обучения (TensorBoard)

```bash
make tensorboard
```

Откройте в браузере: http://localhost:6006

## 🛠 Использование Makefile

### Основные команды

```bash
make help              # Показать все доступные команды
make build             # Собрать Docker образ
make train             # Обучить модель
make inference         # Запустить предсказания
make tensorboard       # Запустить TensorBoard
make jupyter          # Запустить Jupyter Lab
make shell            # Открыть bash в контейнере
make check-gpu        # Проверить GPU
make stop             # Остановить все контейнеры
make clean            # Удалить контейнеры и образы
```

### Продвинутые команды

```bash
make train-resume     # Продолжить обучение с чекпоинта
make evaluate         # Оценить модель
make logs             # Показать логи обучения
make status           # Статус контейнеров
```

## ⚙️ Конфигурация

Основная конфигурация находится в `config.yaml`.

### Основные параметры

```yaml
# Пути
paths:
  train_dir: '/workspace/data/train'
  val_dir: '/workspace/data/val'
  model_save_path: '/workspace/models/plant_disease_resnet50.pth'

# Модель
model:
  name: 'resnet50'  # resnet50, resnet101
  num_classes: 38
  dropout: 0.5

# Обучение
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
```

### Изменение конфигурации

Отредактируйте `config.yaml` и перезапустите обучение:
```bash
make train
```

## 🎓 Обучение модели

### Базовое обучение

```bash
make train
```

### С использованием Docker Compose

```bash
docker-compose up train
```

### Интерактивный режим

```bash
make shell
python3 src/train.py --config config.yaml
```

### Продолжение обучения

```bash
make train-resume
```

### Мониторинг в реальном времени

В отдельном терминале:
```bash
make tensorboard
```

Затем откройте http://localhost:6006

## 🔮 Инференс

### Предсказание для одного изображения

```python
from src.inference import PlantDiseasePredictor

predictor = PlantDiseasePredictor('models/plant_disease_resnet50.pth')
result = predictor.predict('data/test/image.jpg')

print(f"Класс: {result['top_prediction']['class']}")
print(f"Уверенность: {result['top_prediction']['percentage']}")
```

### Batch предсказание

```bash
make inference
```

## 📊 Оценка модели

```bash
make evaluate
```

Результаты будут сохранены в `results/evaluation/`

## 🔥 Jupyter Lab

Запуск Jupyter для экспериментов:

```bash
make jupyter
```

Откройте http://localhost:8888

## 🐛 Решение проблем

### GPU не обнаружен

```bash
# Проверьте NVIDIA драйвер
nvidia-smi

# Проверьте NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Ошибка памяти GPU

Уменьшите `batch_size` в `config.yaml`:
```yaml
training:
  batch_size: 16  # Вместо 32
```

### Медленное обучение

1. Включите mixed precision в `config.yaml`:
```yaml
gpu:
  mixed_precision: true
```

2. Увеличьте `num_workers`:
```yaml
training:
  num_workers: 8
```

## 📈 Результаты

После обучения результаты будут в:
- `models/plant_disease_resnet50.pth` - обученная модель
- `results/logs/` - TensorBoard логи
- `results/checkpoints/` - чекпоинты обучения

## 🤝 Вклад в проект

1. Fork проекта
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

