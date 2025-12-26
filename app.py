# Путь: app.py
"""
Streamlit приложение для детекции огня с помощью обученной модели YOLOv8
"""
# ИМПОРТИРУЕМ ФИКС ПЕРЕД ВСЕМИ ДРУГИМИ ИМПОРТАМИ
from utils.fix_torch_load import apply_torch_load_fix
apply_torch_load_fix()

import streamlit as st
import torch
import warnings
from pathlib import Path
import os
import tempfile
import time
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Подавляем ненужные предупреждения
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

# Настройка страницы
st.set_page_config(
    page_title="🔥 Fire Detector Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пути к проекту
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8_fire.pt"
DEFAULT_MODEL = "yolov8n.pt"

@st.cache_resource
def load_model():
    """Загрузка модели YOLOv8 для детекции огня с кэшированием"""
    print("📥 Загрузка модели для детекции огня...")
    
    # Импортируем Ultralytics ПОСЛЕ применения фиксов
    from ultralytics import YOLO
    
    try:
        if MODEL_PATH.exists():
            print(f"✅ Загрузка обученной модели для детекции огня: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            model_name = "🔥 Обученная модель (Fire Detection)"
        else:
            print(f"⚠️ Модель для детекции огня не найдена по пути {MODEL_PATH}")
            print(f"🔄 Загрузка базовой модели: {DEFAULT_MODEL}")
            model = YOLO(DEFAULT_MODEL)
            model_name = f"📦 Базовая модель ({DEFAULT_MODEL})"
        
        # Тестируем загрузку
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(test_img, verbose=False)
        
        print("✅ Модель успешно загружена и протестирована")
        return model, model_name
    
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели для детекции огня: {str(e)}")
        print("🔄 Попытка загрузить базовую модель...")
        
        try:
            model = YOLO(DEFAULT_MODEL)
            print("✅ Базовая модель загружена успешно")
            return model, f"📦 Базовая модель ({DEFAULT_MODEL})"
        except Exception as e2:
            print(f"❌ Критическая ошибка: {str(e2)}")
            st.error(f"Не удалось загрузить модель: {str(e2)}")
            st.stop()

def process_image(image, model, confidence, iou_threshold):
    """Обработка изображения моделью для детекции огня"""
    start_time = time.time()
    
    # Конвертируем PIL Image в numpy array если нужно
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Выполняем детекцию
    results = model(
        image,
        conf=confidence,
        iou=iou_threshold,
        verbose=False
    )
    
    processing_time = time.time() - start_time
    return results[0], processing_time

def plot_results(image, results):
    """Визуализация результатов детекции огня"""
    # Получаем изображение с bounding boxes
    plotted_img = results.plot()
    
    # Конвертируем BGR в RGB
    if plotted_img.shape[2] == 3:  # Если цветное изображение
        plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
    
    return plotted_img

def main():
    """Основная функция приложения"""
    # Загружаем модель
    model, model_name = load_model()
    
    # SIDEBAR
    with st.sidebar:
        st.title("🔥 Fire Detector Pro")
        st.markdown("---")
        
        # Информация о модели
        st.markdown("### 🧠 Модель")
        st.info(f"**Текущая модель:** {model_name}")
        
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / 1024 / 1024
            st.success(f"✅ Модель загружена ({file_size:.1f} MB)")
        else:
            st.warning("⚠️ Обученная модель не найдена")
        
        st.markdown("---")
        
        # Настройки детекции
        st.markdown("### ⚙️ Настройки детекции")
        confidence = st.slider("Порог уверенности", 0.1, 1.0, 0.4, 0.05,
                              help="Минимальная уверенность для отображения детекции огня")
        iou_threshold = st.slider("Порог IOU", 0.1, 1.0, 0.45, 0.05,
                                help="Порог для подавления перекрывающихся bounding boxes")
        
        # Дополнительные настройки
        st.markdown("---")
        st.markdown("### 📊 Дополнительно")
        show_confidence = st.checkbox("Показывать уверенность", value=True)
        
        st.markdown("---")
        
        # Информация о датасете
        if st.button("ℹ️ О датасете огня"):
            st.markdown("""
            **Датасет: Fire Detection**
            - **Источник:** [Roboflow Universe](https://universe.roboflow.com/sean-cftrp/fire-z2n21)
            - **Изображения:** 6386 (train/val/test)
            - **Train set:** 5580 изображений (87%)
            - **Valid set:** 578 изображений (9%)
            - **Test set:** 228 изображений (4%)
            - **Классы:** fire
            - **Аннотации:** Bounding Boxes
            - **Лицензия:** CC BY 4.0
            
            **Характеристики:**
            - Различные условия освещения
            - Разные масштабы огня
            - Реальные сценарии пожаров
            - Изображения с дымом и пламенем
            """)
        
        st.markdown("---")
        st.markdown("### 💡 Советы по детекции огня")
        st.markdown("""
        - Используйте изображения хорошего качества
        - Модель лучше детектирует яркое пламя
        - Дым может быть сложнее для детекции
        - При слабом освещении уменьшите порог уверенности
        """)

    # MAIN CONTENT
    st.title("🔥 Система детекции огня в реальном времени")
    st.markdown("#### Загрузите изображение для обнаружения огня и пожаров")
    
    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение...", 
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
        help="Поддерживаются форматы: JPG, JPEG, PNG, BMP, WEBP"
    )
    
    # Обработка загруженного файла
    if uploaded_file is not None:
        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Загружаем изображение
            image = Image.open(uploaded_file)
            image = image.convert('RGB')  # Конвертируем в RGB если нужно
            
            # Отображаем оригинальное изображение
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Оригинальное изображение")
                st.image(image, use_column_width=True)
                st.caption(f"Размер: {image.width}×{image.height} пикселей")
            
            # Выполняем детекцию
            with col2:
                st.subheader("🎯 Результат детекции огня")
                
                with st.spinner("🔥 Анализ изображения на наличие огня..."):
                    results, processing_time = process_image(
                        image, 
                        model, 
                        confidence, 
                        iou_threshold
                    )
                    
                    # Визуализируем результаты
                    result_image = plot_results(np.array(image), results)
                    
                    # Отображаем результат
                    st.image(
                        result_image,
                        caption=f"Обнаружено очагов огня: {len(results.boxes)}",
                        use_column_width=True
                    )
            
            # Статистика и информация
            st.markdown("---")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                fire_count = len(results.boxes)
                if fire_count > 0:
                    st.metric("🔥 Обнаружено огня", fire_count)
                else:
                    st.metric("✅ Безопасно", "Нет огня")
            
            with stats_col2:
                if len(results.boxes) > 0:
                    confidences = results.boxes.conf.cpu().numpy()
                    avg_conf = confidences.mean() * 100
                    st.metric("⭐ Средняя уверенность", f"{avg_conf:.1f}%")
            
            with stats_col3:
                st.metric("⚡ Время обработки", f"{processing_time*1000:.1f} мс")
            
            # Детальная информация
            if len(results.boxes) > 0:
                st.markdown("### 📋 Детали детекции огня")
                
                # Подготовка данных для таблицы
                boxes_data = []
                for i, box in enumerate(results.boxes):
                    conf = float(box.conf[0]) * 100
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    class_name = "fire"  # Фиксированный класс
                    
                    # Определяем уровень опасности
                    danger_level = "🔴 Высокая" if conf > 70 else "🟠 Средняя" if conf > 40 else "🟡 Низкая"
                    
                    boxes_data.append({
                        "№": i + 1,
                        "Класс": class_name,
                        "Уверенность (%)": f"{conf:.1f}",
                        "Уровень опасности": danger_level,
                        "Координаты": f"({coords[0]}, {coords[1]}) - ({coords[2]}, {coords[3]})",
                        "Площадь (px²)": f"{(coords[2] - coords[0]) * (coords[3] - coords[1])}"
                    })
                
                # Отображение таблицы
                st.table(boxes_data)
                
                # График распределения уверенности
                if show_confidence:
                    confidences = [float(box.conf[0]) * 100 for box in results.boxes]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(range(len(confidences)), confidences, color=['red' if c > 70 else 'orange' if c > 40 else 'yellow' for c in confidences])
                    ax.set_title('Распределение уверенности по детекциям огня')
                    ax.set_xlabel('Номер детекции')
                    ax.set_ylabel('Уверенность (%)')
                    ax.set_ylim(0, 100)
                    ax.grid(True, alpha=0.3)
                    
                    for i, (bar, conf) in enumerate(zip(bars, confidences)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{conf:.1f}%', ha='center', va='bottom')
                    
                    st.pyplot(fig)
            
            # Рекомендации по безопасности
            st.markdown("---")
            if len(results.boxes) > 0:
                st.warning("🚨 Обнаружен огонь! Рекомендуется принять меры безопасности!")
                st.markdown("""
                ### 📞 Рекомендуемые действия:
                - Немедленно сообщите в пожарную службу: **101** или **112**
                - Эвакуируйте людей из опасной зоны
                - Не пытайтесь тушить огонь самостоятельно, если он крупный
                - Следуйте инструкциям экстренных служб
                """)
            else:
                st.success("✅ Огонь не обнаружен. Ситуация безопасна.")
                st.markdown("""
                ### ✅ Рекомендации:
                - Продолжайте мониторинг ситуации
                - Проверяйте систему детекции регулярно
                - Убедитесь, что датчики дыма работают исправно
                - Соблюдайте правила пожарной безопасности
                """)
            
            # Удаляем временный файл
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
        except Exception as e:
            st.error(f"❌ Ошибка при обработке изображения: {str(e)}")
            st.error("💡 Попробуйте другое изображение или перезагрузите страницу")
    
    else:
        # Информация о загрузке
        st.markdown("### 🎯 Как использовать систему")
        st.markdown("""
        1. Нажмите кнопку **"Browse files"** для загрузки изображения
        2. Выберите файл с изображением в формате JPG, JPEG, PNG, BMP или WEBP
        3. Система автоматически проанализирует изображение на наличие огня
        4. Результаты будут отображены в реальном времени
        
        ### 📸 Примеры изображений для тестирования:
        - Фотографии с открытым пламенем
        - Изображения лесных пожаров
        - Снимки пожаров в помещениях
        - Изображения с дымом (модель может детектировать дым как огонь)
        """)

    # Footer
    st.markdown("---")
    footer_col = st.columns([1, 2, 1])[1]
    
    with footer_col:
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
            "🔥 Fire Detector Pro | YOLOv8 + Streamlit<br>"
            "© 2025 Система пожарной безопасности | "
            "<a href='https://universe.roboflow.com/sean-cftrp/fire-z2n21' "
            "target='_blank' style='color: #cc0000;'>Датасет огня</a>"
            "</div>",
            unsafe_allow_html=True
        )
    
    # Добавляем скрытую информацию для отладки
    with st.expander("🔧 Отладочная информация (для разработчика)"):
        st.markdown(f"**Текущая директория:** `{os.getcwd()}`")
        st.markdown(f"**Путь к модели:** `{MODEL_PATH}`")
        st.markdown(f"**Существует модель:** `{MODEL_PATH.exists()}`")
        if MODEL_PATH.exists():
            st.markdown(f"**Размер модели:** `{MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB`")
        st.markdown(f"**PyTorch версия:** `{torch.__version__}`")
        st.markdown(f"**CUDA доступна:** `{torch.cuda.is_available()}`")
        if torch.cuda.is_available():
            st.markdown(f"**CUDA версия:** `{torch.version.cuda}`")
            st.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")

if __name__ == "__main__":
    main()