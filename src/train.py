import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path


# train.py

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Установка seed для воспроизводимости
        self.seed = self.config.get('seed', 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.paths = self.config['paths']
        self.model_cfg = self.config['model']
        self.training = self.config['training']
        self.augmentation = self.config['augmentation']

        # Устройство
        if self.config['gpu']['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Создание директорий
        os.makedirs(self.paths['results_dir'], exist_ok=True)
        os.makedirs(self.paths['logs_dir'], exist_ok=True)
        os.makedirs(os.path.dirname(self.paths['model_save_path']), exist_ok=True)


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                for img_name in os.listdir(cls_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(cls_path, img_name))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(config, mode='train'):
    aug_cfg = config.augmentation[mode]

    if mode == 'train':
        transform_list = [
            transforms.Resize((aug_cfg['resize'], aug_cfg['resize'])),
        ]

        if 'random_horizontal_flip' in aug_cfg:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=aug_cfg['random_horizontal_flip'])
            )

        if 'random_vertical_flip' in aug_cfg:
            transform_list.append(
                transforms.RandomVerticalFlip(p=aug_cfg['random_vertical_flip'])
            )

        if 'random_rotation' in aug_cfg:
            transform_list.append(
                transforms.RandomRotation(aug_cfg['random_rotation'])
            )

        if 'color_jitter' in aug_cfg:
            cj = aug_cfg['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj['brightness'],
                    contrast=cj['contrast'],
                    saturation=cj['saturation'],
                    hue=cj['hue']
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                aug_cfg['normalize']['mean'],
                aug_cfg['normalize']['std']
            )
        ])
    else:
        transform_list = [
            transforms.Resize((aug_cfg['resize'], aug_cfg['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(
                aug_cfg['normalize']['mean'],
                aug_cfg['normalize']['std']
            )
        ]

    return transforms.Compose(transform_list)


def create_model(config):
    model_name = config.model_cfg['name']
    num_classes = config.model_cfg['num_classes']
    pretrained = config.model_cfg['pretrained']

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config.model_cfg['dropout']),
            nn.Linear(num_features, num_classes)
        )
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config.model_cfg['dropout']),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Заморозка backbone если требуется
    if config.model_cfg['freeze_backbone']:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            # Новая autocast API
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss / len(dataloader):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / len(dataloader):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    return running_loss / len(dataloader), 100. * correct / total


def main(args):
    # Загрузка конфигурации
    config = Config(args.config)
    print(f"Используется устройство: {config.device}")

    # TensorBoard
    writer = None
    if config.config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir=config.paths['logs_dir'])

    # Создание датасетов
    print("Загрузка данных...")
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')

    train_dataset = PlantDiseaseDataset(config.paths['train_dir'], train_transform)
    val_dataset = PlantDiseaseDataset(config.paths['val_dir'], val_transform)

    print(f"Тренировочных изображений: {len(train_dataset)}")
    print(f"Валидационных изображений: {len(val_dataset)}")
    print(f"Количество классов: {len(train_dataset.classes)}")

    # Обновление num_classes в конфиге
    config.model_cfg['num_classes'] = len(train_dataset.classes)

    # ---------------------------
    # БЕЗОПАСНЫЕ DataLoader'ы
    # ---------------------------
    # Причина: у тебя был краш в воркерах из-за /dev/shm и "No space left on device".
    # Решение: num_workers=0 => без форка процессов => не нужен shared memory.
    # pin_memory можно оставить True только если cuda и это не вызывает проблем.
    safe_num_workers = 0  # критично для предотвращения bus error
    pin_mem = (config.device.type == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=safe_num_workers,
        pin_memory=pin_mem,
        persistent_workers=False  # важно при num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=safe_num_workers,
        pin_memory=pin_mem,
        persistent_workers=False
    )

    # Создание модели
    print("Создание модели...")
    model = create_model(config)
    model = model.to(config.device)

    # Функция потерь
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.config['loss'].get('label_smoothing', 0.0)
    )

    # Оптимизатор
    if config.config['optimizer']['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training['learning_rate'],
            weight_decay=config.training.get('weight_decay', 0)
        )
    elif config.config['optimizer']['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training['learning_rate'],
            weight_decay=config.training.get('weight_decay', 0)
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training['learning_rate'],
            momentum=config.config['optimizer'].get('momentum', 0.9),
            weight_decay=config.training.get('weight_decay', 0)
        )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training['scheduler']['factor'],
        patience=config.training['scheduler']['patience'],
        min_lr=config.training['scheduler']['min_lr']
    )

    # Mixed Precision
    scaler = None
    use_amp = config.config['gpu'].get('mixed_precision', False) and torch.cuda.is_available()
    if use_amp:
        # Новая API: вместо torch.cuda.amp.GradScaler()
        scaler = torch.amp.GradScaler('cuda')
        print("Используется Mixed Precision Training")

    # Обучение
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(config.training['num_epochs']):
        print(f"\n{'=' * 60}")
        print(f"Эпоха {epoch + 1}/{config.training['num_epochs']}")
        print(f"{'=' * 60}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device, scaler, use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)

        print(f"\nРезультаты:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # TensorBoard логирование
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': train_dataset.classes,
                'config': config.config
            }, config.paths['model_save_path'])
            print(f"✓ Модель сохранена! Лучшая точность: {best_val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if config.training['early_stopping']['enabled']:
            if patience_counter >= config.training['early_stopping']['patience']:
                print(f"\nEarly stopping на эпохе {epoch + 1}")
                break

    if writer:
        writer.close()

    print(f"\n{'=' * 60}")
    print(f"Обучение завершено!")
    print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    args = parser.parse_args()

    main(args)
