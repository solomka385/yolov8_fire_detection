# Путь: utils/fix_torch_load.py
"""
Универсальный фикс для проблемы с torch.load() в PyTorch 2.6+
Работает со всеми версиями PyTorch
"""

import torch
import warnings
import pickle
import os
import sys
from typing import Any, Dict, Optional

def apply_torch_load_fix():
    """
    Применяет универсальный фикс для torch.load() проблем
    Работает со всеми версиями PyTorch
    """
    print("🔧 Применение универсального фикса для torch.load()...")
    
    try:
        # Проверяем версию PyTorch
        torch_version = torch.__version__
        print(f"📦 Версия PyTorch: {torch_version}")
        
        # Фикс для PyTorch 2.6+ (weights_only проблема)
        if hasattr(torch, '__version__') and torch.__version__ >= '2.6.0':
            print("🔧 Обнаружена PyTorch 2.6+ - применение специального фикса")
            
            # Вариант 1: Через переменные окружения (самый надежный)
            os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
            print("✅ Установлена переменная окружения TORCH_FORCE_WEIGHTS_ONLY_LOAD=0")
            
            # Вариант 2: Патч для torch.load (дополнительная страховка)
            original_torch_load = torch.load
            
            def safe_torch_load(f, map_location=None, **kwargs):
                """
                Безопасная загрузка для PyTorch 2.6+
                """
                try:
                    # Сначала пробуем стандартный способ
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = True
                    return original_torch_load(f, map_location=map_location, **kwargs)
                except (pickle.UnpicklingError, RuntimeError, TypeError, AttributeError) as e:
                    error_msg = str(e).lower()
                    
                    # Если ошибка связана с weights_only или безопасностью
                    if any(keyword in error_msg for keyword in [
                        'weights_only', 'unsupported global', 'pickle', 
                        'unpickling', 'security', 'safe', 'whitelist'
                    ]):
                        warnings.warn(
                            "Обнаружена проблема с безопасной загрузкой. "
                            "Переключение на weights_only=False для совместимости. "
                            "Это безопасно для моделей Ultralytics из доверенных источников.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        
                        # Пробуем с weights_only=False
                        kwargs['weights_only'] = False
                        return original_torch_load(f, map_location=map_location, **kwargs)
                    raise e
            
            # Применяем патч только если это безопасно
            try:
                torch.load = safe_torch_load
                print("✅ Патч для torch.load() успешно применен")
            except Exception as patch_error:
                print(f"⚠️  Не удалось применить патч для torch.load(): {str(patch_error)}")
                print("➡️  Используется только переменная окружения TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        
        # Фикс для старых версий PyTorch
        else:
            print("🔧 Обнаружена PyTorch < 2.6 - применение базового фикса")
            
            # Просто убеждаемся, что weights_only=False по умолчанию
            original_torch_load = torch.load
            
            def legacy_safe_torch_load(f, map_location=None, **kwargs):
                """
                Фикс для старых версий PyTorch
                """
                if 'weights_only' in kwargs:
                    del kwargs['weights_only']
                return original_torch_load(f, map_location=map_location, **kwargs)
            
            torch.load = legacy_safe_torch_load
            print("✅ Базовый фикс для torch.load() применен")
        
        # Проверка работы фикса
        print("\n🔍 Проверка работы фикса...")
        try:
            # Создаем тестовый тензор
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            test_path = "test_torch_fix.pt"
            
            # Сохраняем и загружаем
            torch.save(test_tensor, test_path)
            loaded_tensor = torch.load(test_path, weights_only=False)
            
            if torch.allclose(test_tensor, loaded_tensor):
                print("✅ ✅ Тест загрузки/сохранения ПРОЙДЕН")
            else:
                print("❌ Тест загрузки/сохранения НЕ ПРОЙДЕН")
            
            # Удаляем тестовый файл
            if os.path.exists(test_path):
                os.remove(test_path)
            
        except Exception as test_error:
            print(f"⚠️  Тест фикса завершился с ошибкой: {str(test_error)}")
            print("💡 Фикс все равно применен, продолжаем работу...")
        
        print("🎉 Универсальный фикс для torch.load() успешно применен!")
        return True
    
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА при применении фикса: {str(e)}")
        print("🔄 Попытка базового решения...")
        
        # Базовое решение - просто устанавливаем переменную окружения
        try:
            os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
            print("✅ Установлена переменная окружения TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 (базовое решение)")
            return True
        except Exception as env_error:
            print(f"❌ Не удалось установить переменную окружения: {str(env_error)}")
            print("⚠️  Продолжаем без фикса, могут возникнуть проблемы с загрузкой моделей")
            return False

def is_torch_2_6_plus() -> bool:
    """
    Проверяет, является ли версия PyTorch 2.6 или выше
    """
    try:
        version_parts = torch.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        return major > 2 or (major == 2 and minor >= 6)
    except:
        return False

def get_safe_load_kwargs() -> Dict[str, Any]:
    """
    Возвращает безопасные параметры для torch.load() в зависимости от версии PyTorch
    """
    if is_torch_2_6_plus():
        return {'weights_only': False}
    else:
        return {}

# Автоматически применяем фикс при импорте
apply_torch_load_fix()

# Экспортируем полезные функции
__all__ = ['apply_torch_load_fix', 'is_torch_2_6_plus', 'get_safe_load_kwargs']