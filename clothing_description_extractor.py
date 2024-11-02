# Установка необходимых библиотек (если нужно, раскомментируйте)
# !pip install transformers torch pillow requests python-telegram-bot

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import requests
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Загрузка модели и процессора
model_name = "liuhaotian/llava-onevision"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)

# Промпт для извлечения одежды
prompt = "List all items of clothing in the image and provide detailed descriptions of each item, such as color, pattern, and style."


# Функция для извлечения описаний элементов одежды на изображении
def get_clothing_descriptions(image_url, prompt):
    # Загрузка изображения
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Подготовка изображения и промпта для модели
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cpu")

    # Генерация описаний
    outputs = model.generate(**inputs)
    descriptions = processor.batch_decode(outputs, skip_special_tokens=True)

    return descriptions[0]


# Функция для отображения результатов в Jupyter Notebook
def display_clothing_items(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    display(image)  # Показываем изображение в Jupyter Notebook
    descriptions = get_clothing_descriptions(image_url, prompt)
    print("Detected clothing items:")
    print(descriptions)


# Настройка и запуск Telegram-бота
def start_bot():
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # Замените на токен вашего бота от BotFather

    # Функция для обработки изображений от пользователя
    def handle_image(update, context):
        # Получаем URL изображения
        file = update.message.photo[-1].get_file()
        image_url = file.file_path
        description = get_clothing_descriptions(image_url, prompt)
        update.message.reply_text(description)

    # Настройка и запуск бота
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.photo, handle_image))

    print("Bot started. Send an image to get clothing descriptions.")
    updater.start_polling()
    updater.idle()


# Пример использования
if __name__ == "__main__":
    # Тестирование в Jupyter Notebook:
    image_url = "https://link-to-your-image.jpg"  # Замените на ссылку на изображение с одеждой
    display_clothing_items(image_url)

    # Запуск Telegram-бота
    start_bot()
