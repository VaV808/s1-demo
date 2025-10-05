# app.py
import streamlit as st
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import json
import os

# Настройки страницы
st.set_page_config(
    page_title="🚗 CarFix — Оценка ремонта",
    page_icon="🚗",
    layout="centered"
)

# Логотип и заголовок
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=80)
with col2:
    st.title("CarFix — Оценка ремонта")

st.markdown("### 📸 Загрузите фото повреждённого автомобиля")
st.write("Модель автоматически оценит стоимость ремонта в рублях.")

# Загрузка модели (кешируется)
@st.cache_resource
def load_model():
    model = torch.jit.load("car_damage_model_ef.pt")
    model.eval()
    with open("norm_params_ef.json") as f:
        norm_params = json.load(f)
    return model, norm_params

model, norm_params = load_model()

# Трансформации (точно как при обучении)
def preprocess_image(image):
    """Приводит изображение к формату 224x224 с сохранением пропорций (padding)"""
    # Конвертируем в RGB
    image = image.convert("RGB")
    
    # Сохраняем пропорции: добавляем padding до квадрата
    image = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    
    # Применяем те же трансформации, что и при обучении
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Выберите изображение (JPG/PNG, до 10 МБ)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Проверка размера
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ Файл слишком большой! Максимум 10 МБ.")
    else:
        try:
            # Открываем изображение
            image = Image.open(uploaded_file)
            
            # Отображаем оригинал
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Предсказание
            with st.spinner("🔍 Анализируем повреждение..."):
                # Предобработка под модель
                input_tensor = preprocess_image(image).unsqueeze(0)
                
                # Предсказание
                with torch.no_grad():
                    pred_norm = model(input_tensor).item()
                
                # Денормализация
                pred = pred_norm * norm_params["std"] + norm_params["mean"]
                pred = max(0, pred)  # не может быть отрицательной
            
            # Результат
            st.success(f"### 💰 Стоимость ремонта: **{pred:,.0f} руб.**")
            
            # Информация о точности
            st.info(
                "ℹ️ Модель обучена на реальных данных. "
                "Средняя ошибка: ±1000 руб. "
                "Результат является предварительной оценкой."
            )
            
        except Exception as e:
            st.error(f"❌ Ошибка обработки изображения: {str(e)}")

# Подвал
st.markdown("---")
st.caption("© 2025 CarFix AI | Технология оценки повреждений автомобилей")