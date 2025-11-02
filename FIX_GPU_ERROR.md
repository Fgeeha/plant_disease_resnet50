# 🔧 Быстрое решение ошибки GPU

## Ваша ошибка:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

---

## ✅ РЕШЕНИЕ (3 простых шага)

### Шаг 1: Проверьте NVIDIA драйвер

```bash
nvidia-smi
```

**Если работает** → переходите к Шагу 2  
**Если не работает** → установите драйвер:

```bash
# Ubuntu
sudo ubuntu-drivers autoinstall
sudo reboot
```

---

### Шаг 2: Установите NVIDIA Container Toolkit

```bash
# Автоматически через Makefile
make install-nvidia-docker

# ИЛИ вручную:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### Шаг 3: Проверьте работу

```bash
make check-gpu
```

Должно показать:
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce ...
```

---

## 🎯 Теперь можно работать!

```bash
make train          # Обучение с GPU
make tensorboard    # Мониторинг
```

---

## 🆘 Если не помогло

### Вариант А: Используйте CPU режим (медленнее, но работает)

```bash
make run-cpu        # Запуск без GPU
make train-cpu      # Обучение на CPU
```

### Вариант Б: Полная диагностика

```bash
chmod +x diagnose.sh
./diagnose.sh
```

Скрипт покажет что именно не работает и как исправить.

---

## 📝 Обновленный Makefile

Я обновил Makefile. Теперь он автоматически определяет наличие GPU и использует соответствующие команды.

**Новые команды:**
- `make run-cpu` - запуск без GPU
- `make train-cpu` - обучение на CPU  
- `make shell-cpu` - shell без GPU
- `make install-nvidia-docker` - автоустановка NVIDIA Docker
- `make check-system` - полная диагностика системы

---

## 🔍 Быстрая диагностика

```bash
# 1. Проверка драйвера
nvidia-smi

# 2. Проверка Docker
docker --version

# 3. Проверка NVIDIA Container Toolkit
nvidia-ctk --version

# 4. Проверка runtime в Docker
docker info | grep -i runtime

# 5. Тест GPU в Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 6. Полная диагностика
./diagnose.sh
```

---

## ✅ Контрольный список

- [ ] nvidia-smi работает
- [ ] nvidia-container-toolkit установлен  
- [ ] Docker перезапущен после установки
- [ ] Пользователь в группе docker
- [ ] `docker run --gpus all` работает
- [ ] `make check-gpu` показывает GPU

---

**Всё готово? Запускайте обучение!** 🚀

```bash
make train
```