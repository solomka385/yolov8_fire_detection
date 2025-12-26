# Путь: scripts/train.py
"""
Скрипт для обучения YOLOv8 на локальном датасете огня
Работает с уже загруженным датасетом, не требует интернет-соединения
"""

# === КРИТИЧЕСКИ ВАЖНЫЙ ФИКС - ДОЛЖЕН БЫТЬ ПЕРВЫМ ИМПОРТОМ ===
print("🚀 Запуск скрипта обучения для локального датасета огня...")
print("🔧 Применение фиксов перед импортом других библиотек...")

# Попытка импорта фикса из корневой директории utils
try:
    from utils.fix_torch_load import apply_torch_load_fix, get_safe_load_kwargs
    print("✅ Фикс успешно импортирован из корневой директории utils/")
except ImportError as e1:
    print(f"❌ Ошибка импорта из корневой директории: {str(e1)}")
    
    # Попытка импорта из scripts/utils (если пользователь создал там)
    try:
        from scripts.utils.fix_torch_load import apply_torch_load_fix, get_safe_load_kwargs
        print("✅ Фикс успешно импортирован из scripts/utils/")
    except ImportError as e2:
        print(f"❌ Ошибка импорта из scripts/utils: {str(e2)}")
        
        # Создаем минимальный фикс на лету
        print("🔄 Создание минимального фикса на лету...")
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        print("✅ Установлена переменная окружения TORCH_FORCE_WEIGHTS_ONLY_LOAD=0")
        
        def apply_torch_load_fix():
            return True
        
        def get_safe_load_kwargs():
            return {'weights_only': False}

# Применяем фикс
apply_torch_load_fix()
# ===========================================================

# Теперь можно импортировать остальные библиотеки
import os
import sys
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
import warnings
import torch
import yaml

# Подавляем ненужные предупреждения
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="ultralytics")

# Загрузка переменных окружения
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Константы проекта для локального датасета
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATASET_PATH = Path("C:/Users/Solomka/Downloads/Fire.v1i.yolov8")  # Путь к вашему локальному датасету
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "yolov8_fire.pt"
DATA_YAML = LOCAL_DATASET_PATH / "data.yaml"  # Используем ваш существующий data.yaml
RUNS_DIR = PROJECT_ROOT / "runs"

# Создаем необходимые директории
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🔥 YOLOv8 FIRE DETECTOR - СКРИПТ ОБУЧЕНИЯ (ЛОКАЛЬНЫЙ ДАТАСЕТ)")
print("=" * 60)

# Проверяем версии библиотек
print(f"🔧 Версии библиотек:")
print(f"   • Python: {sys.version.split()[0]}")
print(f"   • PyTorch: {torch.__version__}")
print(f"   • CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   • CUDA версия: {torch.version.cuda}")
    print(f"   • GPU: {torch.cuda.get_device_name(0)}")

print("\n🔄 Проверка импорта Ultralytics...")
try:
    from ultralytics import YOLO
    print("✅ Ultralytics успешно импортирован")
except Exception as e:
    print(f"❌ Ошибка импорта Ultralytics: {str(e)}")
    print("💡 Попробуйте: pip install --upgrade ultralytics")
    sys.exit(1)

def verify_local_dataset():
    """Проверяет структуру локального датасета"""
    print("\n🔍 Проверка структуры локального датасета огня...")
    
    if not LOCAL_DATASET_PATH.exists():
        print(f"❌ Директория датасета не существует: {LOCAL_DATASET_PATH}")
        return False
    
    # Проверяем наличие обязательных директорий
    required_dirs = [
        LOCAL_DATASET_PATH / "train",
        LOCAL_DATASET_PATH / "valid",
        LOCAL_DATASET_PATH / "test"
    ]
    
    all_exists = True
    for dir_path in required_dirs:
        if dir_path.exists():
            images_dir = dir_path / "images"
            labels_dir = dir_path / "labels"
            
            if images_dir.exists() and labels_dir.exists():
                print(f"✅ {dir_path.name} существует и содержит images/ и labels/")
            else:
                print(f"❌ {dir_path.name} существует, но отсутствуют поддиректории images/ или labels/")
                all_exists = False
        else:
            print(f"❌ {dir_path.name} не существует")
            all_exists = False
    
    # Проверяем наличие data.yaml
    if DATA_YAML.exists():
        print(f"✅ data.yaml найден: {DATA_YAML}")
        
        # Показываем содержимое data.yaml для проверки
        with open(DATA_YAML, 'r') as f:
            content = f.read()
            print("\n📄 Содержимое data.yaml:")
            print(content)
        
        # Проверяем, что классы правильно указаны
        if "names" not in content or "fire" not in content.lower():
            print("⚠️  Внимание: В data.yaml указано некорректное название класса")
            print("🔄  Исправляем конфигурацию для правильной работы...")
            fix_data_yaml()
            return True
        
        return True
    else:
        print(f"❌ data.yaml не найден по пути: {DATA_YAML}")
        return False

def fix_data_yaml():
    """Исправляет data.yaml для правильной работы с YOLOv8"""
    print("\n🔧 Исправление data.yaml для детекции огня...")
    
    try:
        # Читаем текущий YAML файл
        with open(DATA_YAML, 'r') as f:
            data = yaml.safe_load(f)
        
        print("📊 Текущая конфигурация:")
        print(f"   • train: {data.get('train', 'не указан')}")
        print(f"   • val: {data.get('val', 'не указан')}")
        print(f"   • test: {data.get('test', 'не указан')}")
        print(f"   • nc: {data.get('nc', 'не указан')}")
        print(f"   • names: {data.get('names', 'не указан')}")
        
        # Исправляем конфигурацию
        fixed_config = {
            'path': str(LOCAL_DATASET_PATH),  # Путь к корню датасета
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['fire'],  # Правильное название класса
            'roboflow': {
                'workspace': 'sean-cftrp',
                'project': 'fire-z2n21',
                'version': 1,
                'license': 'CC BY 4.0',
                'url': 'https://universe.roboflow.com/sean-cftrp/fire-z2n21/dataset/1'
            }
        }
        
        # Сохраняем исправленный файл
        backup_path = DATA_YAML.with_suffix('.yaml.bak')
        if not backup_path.exists():
            shutil.copy2(DATA_YAML, backup_path)
            print(f"💾 Создана резервная копия: {backup_path}")
        
        with open(DATA_YAML, 'w') as f:
            yaml.dump(fixed_config, f, default_flow_style=False)
        
        print("✅ Конфигурация успешно исправлена!")
        print("📄 Новое содержимое data.yaml:")
        with open(DATA_YAML, 'r') as f:
            print(f.read())
        
        return True
    
    except Exception as e:
        print(f"❌ Ошибка при исправлении data.yaml: {str(e)}")
        print("💡 Ручное исправление:")
        print("   1. Откройте файл C:/Users/Solomka/Downloads/Fire.v1i.yolov8/data.yaml")
        print("   2. Замените содержимое на:")
        print("""
path: C:/Users/Solomka/Downloads/Fire.v1i.yolov8
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['fire']

roboflow:
  workspace: sean-cftrp
  project: fire-z2n21
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/sean-cftrp/fire-z2n21/dataset/1
        """)
        return False

def train_model():
    """Обучение модели YOLOv8 для детекции огня на локальном датасете"""
    print("\n🚀 Начало обучения модели YOLOv8 для детекции огня...")
    
    try:
        # Очищаем кэш CUDA если есть
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Очищен кэш CUDA")
        
        # Загружаем предобученную модель с безопасными параметрами
        print("📥 Загрузка предобученной модели yolov8n.pt...")
        safe_kwargs = get_safe_load_kwargs()
        print(f"🔒 Параметры безопасной загрузки: {safe_kwargs}")
        
        model = YOLO("yolov8n.pt")
        
        print("✅ Модель успешно загружена")
        
        # Проверяем, что файл конфигурации существует
        if not DATA_YAML.exists():
            print(f"❌ Файл конфигурации не существует: {DATA_YAML}")
            print("💡  Убедитесь, что вы правильно указали путь к датасету")
            print(f"    Текущий путь: {LOCAL_DATASET_PATH}")
            return False
        
        # Параметры обучения для датасета огня
        training_params = {
            'data': str(DATA_YAML),
            'epochs': 50,
            'imgsz': 640,
            'batch': 8,
            'name': 'fire_training',
            'patience': 10,
            'device': '0',  # CPU для максимальной совместимости
            'workers': 4,
            'cache': True,  # Кэширование изображений в памяти
            'amp': True,    # Автоматическая смешанная точность
            'exist_ok': True,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'val': True,
            'save': True,
            'save_period': 5,
        }
        
        print("🎯 Параметры обучения для детекции огня:")
        for key, value in training_params.items():
            print(f"  • {key}: {value}")
        
        # Обучение модели
        print("\n⚡ Начало обучения для детекции огня...")
        start_time = time.time()
        
        results = model.train(**training_params)
        
        training_time = time.time() - start_time
        print(f"✅ Обучение завершено за {training_time/60:.1f} минут")
        
        # Ищем лучшие веса
        best_weights_path = None
        search_paths = [
            RUNS_DIR / "detect" / "fire_training" / "weights" / "best.pt",
            Path("runs") / "detect" / "fire_training" / "weights" / "best.pt",
            Path("runs/detect/fire_training/weights/best.pt")
        ]
        
        for path in search_paths:
            if path.exists():
                best_weights_path = path
                break
        
        if best_weights_path and best_weights_path.exists():
            print(f"🏆 Найдены лучшие веса для детекции огня: {best_weights_path}")
            
            # Копируем лучшие веса
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_weights_path, MODEL_PATH)
            print(f"💾 Модель для детекции огня сохранена: {MODEL_PATH}")
            
            return True
        else:
            print(f"❌ Лучшие веса не найдены. Поисковые пути:")
            for path in search_paths:
                print(f"   • {path} - {'найден' if path.exists() else 'не найден'}")
            
            # Показываем содержимое runs директории для отладки
            print("\n📁 Содержимое директории runs:")
            if RUNS_DIR.exists():
                for item in RUNS_DIR.glob("**/*"):
                    if item.is_file():
                        print(f"   • {item.relative_to(RUNS_DIR)}")
            else:
                print("   ❌ Директория runs не существует")
            
            return False
    
    except Exception as e:
        print(f"❌ Ошибка при обучении модели для детекции огня: {str(e)}")
        print("\n💡 Рекомендации по решению проблемы:")
        print("   1. Убедитесь, что датасет загружен правильно")
        print("   2. Проверьте, что файл data.yaml существует и корректен")
        print("   3. Попробуйте уменьшить batch size до 4")
        print("   4. Убедитесь, что достаточно места на диске")
        print("   5. Проверьте логи в директории runs/")
        
        # Дополнительная отладка
        print("\n🔍 Дополнительная отладка:")
        print(f"   • Текущая директория: {os.getcwd()}")
        print(f"   • Путь к data.yaml: {DATA_YAML}")
        print(f"   • Существует data.yaml: {DATA_YAML.exists()}")
        if DATA_YAML.exists():
            file_size = DATA_YAML.stat().st_size
            print(f"   • Размер data.yaml: {file_size} байт")
            with open(DATA_YAML, 'r') as f:
                print(f"   • Содержимое data.yaml: {f.read()[:100]}...")
        
        return False

def main():
    """Основная функция обучения"""
    
    # Шаг 1: Проверка датасета
    print("\n" + "-" * 60)
    print("🔥 ШАГ 1: ПРОВЕРКА ЛОКАЛЬНОГО ДАТАСЕТА ОГНЯ")
    print("-" * 60)
    
    dataset_ok = verify_local_dataset()
    if not dataset_ok:
        print("❌ Не удалось проверить или исправить датасет. Завершение работы.")
        return
    
    # Шаг 2: Обучение модели
    print("\n" + "-" * 60)
    print("🚀 ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ДЕТЕКЦИИ ОГНЯ")
    print("-" * 60)
    
    if MODEL_PATH.exists():
        user_input = input(f"🚨 Модель для детекции огня уже существует по пути {MODEL_PATH}. Перезаписать? (y/n): ").strip().lower()
        if user_input != 'y':
            print("⏭️  Обучение пропущено. Используем существующую модель.")
            return
    
    success = train_model()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ОБУЧЕНИЕ ДЛЯ ДЕТЕКЦИИ ОГНЯ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)
        print(f"📁 Модель для детекции огня сохранена: {MODEL_PATH.absolute()}")
        print(f"📊 Для запуска приложения выполните: streamlit run app.py")
        
        # Проверка существования модели
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / 1024 / 1024
            print(f"✅ ✅ Файл модели существует! Размер: {file_size:.1f} MB")
        else:
            print(f"❌ Файл модели НЕ СУЩЕСТВУЕТ по пути: {MODEL_PATH.absolute()}")
    else:
        print("\n" + "=" * 60)
        print("❌ ОБУЧЕНИЕ ДЛЯ ДЕТЕКЦИИ ОГНЯ ЗАВЕРШЕНО С ОШИБКАМИ")
        print("=" * 60)
        print("💡 Что можно сделать:")
        print("   • Проверить логи в директории runs/")
        print("   • Уменьшить batch size в параметрах обучения")
        print("   • Убедиться, что датасет загружен правильно")
        print("   • Попробовать обучить на меньшем количестве эпох")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        print("💡 Попробуйте следующие шаги:")
        print("   1. Проверьте правильность пути к датасету")
        print("   2. Убедитесь, что у вас достаточно места на диске")
        print("   3. Обновите зависимости: pip install -r requirements.txt --upgrade")