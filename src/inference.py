import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json


class PlantDiseasePredictor:
    def __init__(self, model_path='plant_disease_resnet50.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка чекпоинта
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        num_classes = len(self.classes)

        # Создание модели
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

        # Загрузка весов
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"Модель загружена. Точность на валидации: {checkpoint['val_acc']:.2f}%")
        print(f"Количество классов: {num_classes}")

    def predict(self, image_path, top_k=5):
        """
        Предсказание для одного изображения

        Args:
            image_path: путь к изображению
            top_k: количество топ предсказаний

        Returns:
            dict: словарь с предсказаниями и вероятностями
        """
        # Загрузка и обработка изображения
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Предсказание
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Получение топ-k предсказаний
        top_probs, top_indices = probabilities.topk(top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                'class': self.classes[idx],
                'probability': float(prob),
                'percentage': f"{prob * 100:.2f}%"
            })

        return {
            'top_prediction': results[0],
            'all_predictions': results
        }

    def predict_batch(self, image_paths):
        """Предсказание для нескольких изображений"""
        results = {}
        for img_path in image_paths:
            try:
                results[img_path] = self.predict(img_path)
            except Exception as e:
                results[img_path] = {'error': str(e)}
        return results


# Пример использования
if __name__ == '__main__':
    # Инициализация предиктора
    predictor = PlantDiseasePredictor('plant_disease_resnet50.pth')

    # Предсказание для одного изображения
    image_path = 'test_image.jpg'
    result = predictor.predict(image_path, top_k=3)

    print(f"\n=== Результаты для {image_path} ===")
    print(f"Основное предсказание: {result['top_prediction']['class']}")
    print(f"Уверенность: {result['top_prediction']['percentage']}")

    print("\nВсе топ предсказания:")
    for i, pred in enumerate(result['all_predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['percentage']}")

    # Сохранение результатов в JSON
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)