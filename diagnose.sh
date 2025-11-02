#!/bin/bash

# diagnose.sh - Диагностика системы для Plant Disease Detector

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║       System Diagnostic Tool                          ║"
echo "║       Plant Disease Detector                          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

check_success=0
check_warning=0
check_error=0

print_check() {
    local status=$1
    local message=$2

    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
        ((check_success++))
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
        ((check_warning++))
    else
        echo -e "${RED}✗${NC} $message"
        ((check_error++))
    fi
}

# Проверка 1: Операционная система
echo -e "${BLUE}[1/10] Операционная система${NC}"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_NAME=$(lsb_release -d 2>/dev/null | cut -f2 || echo "Linux")
    print_check "OK" "OS: $OS_NAME"
else
    print_check "WARN" "OS: $OSTYPE (может не поддерживаться)"
fi
echo ""

# Проверка 2: Docker
echo -e "${BLUE}[2/10] Docker${NC}"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
    print_check "OK" "Docker установлен: $DOCKER_VERSION"

    # Проверка прав
    if docker ps &> /dev/null; then
        print_check "OK" "Docker permissions: OK"
    else
        print_check "ERROR" "Docker permissions: Нужен sudo или добавьте пользователя в группу docker"
        echo "  Решение: sudo usermod -aG docker \$USER && newgrp docker"
    fi
else
    print_check "ERROR" "Docker не установлен"
    echo "  Решение: https://docs.docker.com/get-docker/"
fi
echo ""

# Проверка 3: NVIDIA Driver
echo -e "${BLUE}[3/10] NVIDIA Driver${NC}"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)

    print_check "OK" "NVIDIA Driver: $DRIVER_VERSION"
    print_check "OK" "GPU: $GPU_NAME"
    print_check "OK" "GPU Memory: $GPU_MEMORY"
else
    print_check "WARN" "NVIDIA Driver не найден"
    echo "  Если у вас NVIDIA GPU, установите драйвер:"
    echo "  Ubuntu: sudo ubuntu-drivers autoinstall && sudo reboot"
fi
echo ""

# Проверка 4: NVIDIA Container Toolkit
echo -e "${BLUE}[4/10] NVIDIA Container Toolkit${NC}"
if command -v nvidia-ctk &> /dev/null; then
    NVIDIA_CTK_VERSION=$(nvidia-ctk --version 2>&1 | grep version | awk '{print $3}')
    print_check "OK" "NVIDIA Container Toolkit: $NVIDIA_CTK_VERSION"
else
    print_check "ERROR" "NVIDIA Container Toolkit не установлен"
    echo "  Решение: make install-nvidia-docker"
    echo "  Или см.: NVIDIA_DOCKER_SETUP.md"
fi
echo ""

# Проверка 5: Docker NVIDIA Runtime
echo -e "${BLUE}[5/10] Docker NVIDIA Runtime${NC}"
if docker info 2>/dev/null | grep -q nvidia; then
    print_check "OK" "NVIDIA runtime настроен в Docker"
else
    print_check "ERROR" "NVIDIA runtime не найден в Docker"
    echo "  Решение: sudo nvidia-ctk runtime configure --runtime=docker"
    echo "           sudo systemctl restart docker"
fi
echo ""

# Проверка 6: GPU в Docker
echo -e "${BLUE}[6/10] GPU доступность в Docker${NC}"
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_check "OK" "GPU доступен в Docker контейнерах"
else
    print_check "ERROR" "GPU недоступен в Docker"
    echo "  Это критическая ошибка. См. NVIDIA_DOCKER_SETUP.md"
fi
echo ""

# Проверка 7: Docker образ проекта
echo -e "${BLUE}[7/10] Docker образ проекта${NC}"
if docker images | grep -q plant-disease-detector; then
    IMAGE_SIZE=$(docker images plant-disease-detector --format "{{.Size}}")
    print_check "OK" "Образ plant-disease-detector существует ($IMAGE_SIZE)"
else
    print_check "WARN" "Образ не собран"
    echo "  Решение: make build"
fi
echo ""

# Проверка 8: Структура директорий
echo -e "${BLUE}[8/10] Структура проекта${NC}"
REQUIRED_DIRS=("data" "models" "results" "src")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_check "OK" "Директория $dir/ существует"
    else
        print_check "WARN" "Директория $dir/ не найдена"
        echo "  Решение: make setup"
    fi
done
echo ""

# Проверка 9: Конфигурационные файлы
echo -e "${BLUE}[9/10] Конфигурационные файлы${NC}"
CONFIG_FILES=("Dockerfile" "Makefile" "config.yaml" "requirements.txt")
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_check "OK" "$file существует"
    else
        print_check "ERROR" "$file не найден"
    fi
done
echo ""

# Проверка 10: Данные
echo -e "${BLUE}[10/10] Данные для обучения${NC}"
if [ -d "data/train" ] && [ "$(ls -A data/train)" ]; then
    TRAIN_CLASSES=$(find data/train -mindepth 1 -maxdepth 1 -type d | wc -l)
    print_check "OK" "Тренировочные данные: $TRAIN_CLASSES классов"
else
    print_check "WARN" "Тренировочные данные не найдены"
    echo "  Решение: Подготовьте данные или make download-plantvillage"
fi

if [ -d "data/val" ] && [ "$(ls -A data/val)" ]; then
    VAL_CLASSES=$(find data/val -mindepth 1 -maxdepth 1 -type d | wc -l)
    print_check "OK" "Валидационные данные: $VAL_CLASSES классов"
else
    print_check "WARN" "Валидационные данные не найдены"
fi
echo ""

# Итоги
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    РЕЗУЛЬТАТЫ                         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}✓ Успешно: $check_success${NC}"
echo -e "${YELLOW}⚠ Предупреждения: $check_warning${NC}"
echo -e "${RED}✗ Ошибки: $check_error${NC}"
echo ""

if [ $check_error -eq 0 ]; then
    echo -e "${GREEN}🎉 Система готова к работе!${NC}"
    echo ""
    echo "Следующие шаги:"
    echo "  1. make build          # Собрать образ (если не собран)"
    echo "  2. make train          # Начать обучение"
    echo "  3. make tensorboard    # Мониторинг"
elif [ $check_error -le 2 ]; then
    echo -e "${YELLOW}⚠ Система частично готова${NC}"
    echo ""
    echo "Исправьте ошибки выше или используйте CPU режим:"
    echo "  make run-cpu"
    echo "  make train-cpu"
else
    echo -e "${RED}✗ Обнаружены критические ошибки${NC}"
    echo ""
    echo "Рекомендации:"
    echo "  1. Установите недостающие компоненты"
    echo "  2. См. NVIDIA_DOCKER_SETUP.md для настройки GPU"
    echo "  3. Запустите: ./setup.sh для автоматической настройки"
fi

echo ""
echo -e "${BLUE}Для подробной информации см.:${NC}"
echo "  - README.md"
echo "  - NVIDIA_DOCKER_SETUP.md"
echo "  - QUICKSTART.md"