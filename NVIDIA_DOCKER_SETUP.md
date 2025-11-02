# 🚀 Установка NVIDIA Container Toolkit

## Проблема

Вы видите ошибку:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

Это означает, что **NVIDIA Container Toolkit** не установлен или не настроен.

---

## ✅ Решение (Автоматическое)

### Способ 1: Используя Makefile (Рекомендуется)

```bash
# Установить NVIDIA Container Toolkit
make install-nvidia-docker

# После установки проверить
make check-gpu
```

---

## 🔧 Решение (Ручное)

### Шаг 1: Проверка NVIDIA драйвера

```bash
nvidia-smi
```

Должно показать информацию о вашей видеокарте. Если нет - установите драйверы NVIDIA.

### Шаг 2: Установка NVIDIA Container Toolkit

#### Для Ubuntu/Debian:

```bash
# Добавить репозиторий
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Обновить и установить
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Настроить Docker для использования NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Перезапустить Docker
sudo systemctl restart docker
```

#### Для RHEL/CentOS:

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo yum install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Шаг 3: Проверка установки

```bash
# Тест 1: NVIDIA CUDA базовый образ
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Тест 2: Наш образ
make check-gpu
```

Если всё работает, вы увидите информацию о GPU.

---

## 🔍 Диагностика проблем

### Проблема 1: nvidia-smi не найден

```bash
# Проверить установку драйвера
dpkg -l | grep nvidia-driver

# Установить драйвер (Ubuntu)
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Проблема 2: Docker daemon error

```bash
# Проверить Docker runtime
docker info | grep -i runtime

# Должно быть:
# Runtimes: io.containerd.runc.v2 nvidia runc

# Если нет nvidia, пересоздать конфигурацию
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Проблема 3: Permission denied

```bash
# Добавить пользователя в группу docker
sudo usermod -aG docker $USER
newgrp docker

# Или перезагрузиться
sudo reboot
```

### Проблема 4: Старая версия NVIDIA Docker (nvidia-docker2)

```bash
# Удалить старую версию
sudo apt-get remove nvidia-docker2 nvidia-container-runtime

# Установить новую
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 📋 Проверочный чеклист

```bash
# 1. NVIDIA драйвер
nvidia-smi
# ✅ Должен показать GPU

# 2. Docker версия
docker --version
# ✅ Должен быть >= 19.03

# 3. NVIDIA Container Toolkit
nvidia-ctk --version
# ✅ Должен показать версию

# 4. Docker runtime конфигурация
docker info | grep -i runtime
# ✅ Должен показать nvidia в списке

# 5. Тест CUDA контейнера
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# ✅ Должен показать GPU внутри контейнера

# 6. Тест нашего образа
make check-gpu
# ✅ Должен показать CUDA available: True
```

---

## 🆘 Если ничего не помогло

### Полная переустановка

```bash
# 1. Остановить Docker
sudo systemctl stop docker

# 2. Удалить всё связанное с NVIDIA Docker
sudo apt-get purge -y nvidia-docker2 nvidia-container-runtime nvidia-container-toolkit
sudo apt-get autoremove -y

# 3. Очистить конфигурацию Docker
sudo rm -rf /etc/docker/daemon.json.bak
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak 2>/dev/null || true

# 4. Установить заново
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 5. Настроить
sudo nvidia-ctk runtime configure --runtime=docker

# 6. Запустить Docker
sudo systemctl start docker

# 7. Проверить
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🎯 Альтернатива: Работа без GPU

Если у вас нет NVIDIA GPU или не получается настроить, можно работать на CPU:

```bash
# Использовать CPU версии команд
make run-cpu
make train-cpu
make shell-cpu

# Или изменить config.yaml
gpu:
  use_cuda: false
```

**Внимание:** Обучение на CPU будет значительно медленнее!

---

## 📚 Полезные ссылки

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [NVIDIA Docker GitHub](https://github.com/NVIDIA/nvidia-docker)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)

---

## ✅ После успешной установки

```bash
# Проверьте всё
make check-system
make check-gpu

# Теперь можно обучать!
make train
```

**Удачи! 🚀**