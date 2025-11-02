#!/bin/bash

# setup.sh - Скрипт быстрой установки Plant Disease Detector

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   Plant Disease Detector - Installation Script       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Функция для вывода с цветом
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Проверка ОС
print_info "Проверка операционной системы..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_success "Linux обнаружен"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    print_warning "macOS обнаружен (NVIDIA GPU не поддерживается на Mac)"
else
    print_error "Неподдерживаемая ОС: $OSTYPE"
    exit 1
fi

# Проверка Docker
print_info "Проверка Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker установлен: $DOCKER_VERSION"
else
    print_error "Docker не установлен!"
    print_info "Установите Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Проверка nvidia-smi
print_info "Проверка NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    print_success "NVIDIA GPU обнаружен: $GPU_NAME"
else
    print_warning "nvidia-smi не найден. NVIDIA драйверы не установлены?"
    print_info "Продолжить без GPU? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi

# Проверка NVIDIA Container Toolkit
print_info "Проверка NVIDIA Container Toolkit..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_success "NVIDIA Container Toolkit работает"
else
    print_warning "NVIDIA Container Toolkit не настроен"
    print_info "Установить NVIDIA Container Toolkit? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        print_info "Установка NVIDIA Container Toolkit..."

        # Добавление репозитория
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

        # Установка
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit

        # Настройка Docker
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker

        # Проверка
        sleep 2
        if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            print_success "NVIDIA Container Toolkit успешно установлен"
        else
            print_error "Ошибка установки. Попробуйте вручную: см. NVIDIA_DOCKER_SETUP.md"
        fi
    else
        print_warning "Вы можете использовать CPU режим: make run-cpu, make train-cpu"
    fi
fi

# Проверка Make
print_info "Проверка Make..."
if command -v make &> /dev/null; then
    print_success "Make установлен"
else
    print_warning "Make не установлен"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y make
        print_success "Make установлен"
    fi
fi

# Создание структуры директорий
print_info "Создание структуры директорий..."
mkdir -p data/train data/val data/test
mkdir -p models
mkdir -p results/logs results/checkpoints
mkdir -p notebooks
mkdir -p src

# Создание .gitkeep файлов
touch data/.gitkeep models/.gitkeep results/.gitkeep

print_success "Директории созданы"

# Создание базовой структуры файлов если не существует
if [ ! -f "config.yaml" ]; then
    print_info "Создание config.yaml..."
    # config.yaml должен быть создан вручную или скопирован
    print_warning "Не забудьте создать config.yaml!"
fi

# Проверка Kaggle API для загрузки данных
print_info "Проверка Kaggle API..."
if command -v kaggle &> /dev/null; then
    print_success "Kaggle API установлен"
    if [ -f "$HOME/.kaggle/kaggle.json" ]; then
        print_success "Kaggle credentials найдены"
        print_info "Хотите скачать PlantVillage датасет? (y/n)"
        read -r response
        if [[ "$response" == "y" ]]; then
            print_info "Загрузка датасета (это может занять время)..."
            make download-plantvillage || print_warning "Ошибка загрузки. Скачайте вручную."
        fi
    else
        print_warning "Kaggle credentials не найдены"
        print_info "Создайте ~/.kaggle/kaggle.json для загрузки датасетов"
    fi
else
    print_warning "Kaggle API не установлен"
    print_info "Установите: pip install kaggle"
fi

# Построение Docker образа
print_info "Собрать Docker образ сейчас? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    print_info "Сборка Docker образа..."
    make build
    print_success "Docker образ собран"
fi

# Проверка GPU в Docker
if command -v nvidia-smi &> /dev/null; then
    print_info "Проверка GPU в Docker контейнере..."
    if make check-gpu; then
        print_success "GPU доступен в Docker"
    else
        print_warning "GPU недоступен в Docker"
    fi
fi

# Финальное сообщение
echo -e "\n${GREEN}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║            Установка завершена успешно!               ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "\n${YELLOW}Следующие шаги:${NC}"
echo "1. Подготовьте данные в data/train и data/val"
echo "2. Настройте config.yaml"
echo "3. Запустите обучение: ${GREEN}make train${NC}"
echo "4. Мониторинг: ${GREEN}make tensorboard${NC}"
echo ""
echo "Все команды: ${GREEN}make help${NC}"
echo ""
echo -e "${GREEN}Удачи в обучении! 🚀${NC}"