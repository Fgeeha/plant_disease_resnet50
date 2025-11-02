.PHONY: help build run train inference tensorboard jupyter shell stop clean download-data setup

# Переменные
IMAGE_NAME = plant-disease-detector
CONTAINER_NAME = plant-disease-container
DATA_DIR = $(PWD)/data
MODELS_DIR = $(PWD)/models
RESULTS_DIR = $(PWD)/results

# Цвета для вывода
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

# Проверка наличия NVIDIA Docker
HAS_NVIDIA_DOCKER := $(shell docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi 2>/dev/null && echo "yes" || echo "no")

# Флаги GPU (используем если доступно)
ifeq ($(HAS_NVIDIA_DOCKER),yes)
    GPU_FLAGS = --gpus all
    GPU_RUNTIME = --runtime=nvidia
else
    GPU_FLAGS =
    GPU_RUNTIME =
endif

help: ## Показать справку
	@echo "$(GREEN)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Первоначальная настройка проекта
	@echo "$(GREEN)Создание необходимых директорий...$(NC)"
	mkdir -p data/train data/val models results logs
	@echo "$(GREEN)Настройка завершена!$(NC)"

build: ## Собрать Docker образ
	@echo "$(GREEN)Сборка Docker образа...$(NC)"
	docker build -t $(IMAGE_NAME) .
	@echo "$(GREEN)Образ успешно собран!$(NC)"

build-no-cache: ## Собрать Docker образ без кэша
	@echo "$(GREEN)Сборка Docker образа без кэша...$(NC)"
	docker build --no-cache -t $(IMAGE_NAME) .

run: ## Запустить контейнер в интерактивном режиме
	@echo "$(GREEN)Запуск контейнера...$(NC)"
	docker run --gpus all -it --rm \
		--name $(CONTAINER_NAME) \
		-v $(DATA_DIR):/workspace/data \
		-v $(MODELS_DIR):/workspace/models \
		-v $(RESULTS_DIR):/workspace/results \
		-v $(PWD)/src:/workspace/src \
		$(IMAGE_NAME) /bin/bash

train: ## Обучить модель
	@echo "$(GREEN)Запуск обучения модели...$(NC)"
	docker run --gpus all --rm \
		--name $(CONTAINER_NAME) \
		-v $(DATA_DIR):/workspace/data \
		-v $(MODELS_DIR):/workspace/models \
		-v $(RESULTS_DIR):/workspace/results \
		-v $(PWD)/src:/workspace/src \
		$(IMAGE_NAME) python3 src/train.py

train-resume: ## Продолжить обучение с чекпоинта
	@echo "$(GREEN)Продолжение обучения...$(NC)"
	docker run --gpus all --rm \
		--name $(CONTAINER_NAME) \
		-v $(DATA_DIR):/workspace/data \
		-v $(MODELS_DIR):/workspace/models \
		-v $(RESULTS_DIR):/workspace/results \
		-v $(PWD)/src:/workspace/src \
		$(IMAGE_NAME) python3 src/train.py --resume

inference: ## Запустить предсказание на тестовых данных
	@echo "$(GREEN)Запуск инференса...$(NC)"
	docker run --gpus all --rm \
		--name $(CONTAINER_NAME) \
		-v $(DATA_DIR):/workspace/data \
		-v $(MODELS_DIR):/workspace/models \
		-v $(RESULTS_DIR):/workspace/results \
		-v $(PWD)/src:/workspace/src \
		$(IMAGE_NAME) python3 src/inference.py

evaluate: ## Оценить модель на валидационном наборе
	@echo "$(GREEN)Оценка модели...$(NC)"
	docker run --gpus all --rm \
		--name $(CONTAINER_NAME) \
		-v $(DATA_DIR):/workspace/data \
		-v $(MODELS_DIR):/workspace/models \
		-v $(RESULTS_DIR):/workspace/results \
		-v $(PWD)/src:/workspace/src \
		$(IMAGE_NAME) python3 src/evaluate.py

tensorboard: ## Запустить TensorBoard
	@echo "$(GREEN)Запуск TensorBoard на http://localhost:6006$(NC)"
	docker run --gpus all -d --rm \
		--name tensorboard \
		-p 6006:6006 \
		-v $(RESULTS_DIR):/workspace/results \
		$(IMAGE_NAME) tensorboard --logdir=/workspace/results/logs --host=0.0.0.0

jupyter: ## Запустить Jupyter Lab
	@echo "$(GREEN)Запуск Jupyter Lab на http://localhost:8888$(NC)"
	docker run --gpus all -d --rm \
		--name jupyter \
		-p 8888:8888 \
		-v $(DATA_DIR):/workspace/data \
		-v $(MODELS_DIR):/workspace/models \
		-v $(RESULTS_DIR):/workspace/results \
		-v $(PWD)/notebooks:/workspace/notebooks \
		-v $(PWD)/src:/workspace/src \
		$(IMAGE_NAME) jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

shell: ## Открыть bash в контейнере
	@echo "$(GREEN)Открытие shell в контейнере...$(NC)"
	docker run --gpus all -it --rm \
		--name $(CONTAINER_NAME) \
		-v $(DATA_DIR):/workspace/data \
		-v $(MODELS_DIR):/workspace/models \
		-v $(RESULTS_DIR):/workspace/results \
		-v $(PWD)/src:/workspace/src \
		$(IMAGE_NAME) /bin/bash

check-gpu: ## Проверить доступность GPU
	@echo "$(GREEN)Проверка GPU...$(NC)"
	docker run --gpus all --rm $(IMAGE_NAME) python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

download-plantvillage: ## Скачать PlantVillage датасет (требуется Kaggle API)
	@echo "$(GREEN)Скачивание PlantVillage датасета...$(NC)"
	@echo "Убедитесь, что у вас установлен Kaggle API и настроен ~/.kaggle/kaggle.json"
	kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p $(DATA_DIR)
	unzip $(DATA_DIR)/new-plant-diseases-dataset.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/new-plant-diseases-dataset.zip
	@echo "$(GREEN)Датасет скачан!$(NC)"

stop: ## Остановить все контейнеры
	@echo "$(GREEN)Остановка контейнеров...$(NC)"
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker stop tensorboard 2>/dev/null || true
	-docker stop jupyter 2>/dev/null || true

clean: ## Удалить контейнеры и образы
	@echo "$(GREEN)Очистка...$(NC)"
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "$(GREEN)Очистка завершена!$(NC)"

clean-data: ## ОСТОРОЖНО: Удалить все данные и модели
	@echo "$(GREEN)ВНИМАНИЕ: Это удалит все данные, модели и результаты!$(NC)"
	@read -p "Вы уверены? (y/N): " confirm && [ "$$confirm" = "y" ]
	rm -rf $(DATA_DIR)/* $(MODELS_DIR)/* $(RESULTS_DIR)/*
	@echo "$(GREEN)Данные удалены!$(NC)"

logs: ## Показать логи обучения
	@echo "$(GREEN)Последние логи:$(NC)"
	@tail -n 50 $(RESULTS_DIR)/logs/training.log 2>/dev/null || echo "Логи не найдены"

status: ## Показать статус контейнеров
	@echo "$(GREEN)Статус контейнеров:$(NC)"
	@docker ps -a | grep -E "$(IMAGE_NAME)|CONTAINER" || echo "Контейнеры не запущены"

test: ## Запустить тесты
	@echo "$(GREEN)Запуск тестов...$(NC)"
	docker run --gpus all --rm \
		--name $(CONTAINER_NAME) \
		-v $(PWD)/src:/workspace/src \
		-v $(PWD)/tests:/workspace/tests \
		$(IMAGE_NAME) python3 -m pytest tests/ -v

# Значение по умолчанию
.DEFAULT_GOAL := help