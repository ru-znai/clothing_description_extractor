
---

# Clothing Description Extractor

Этот проект использует Vision-Language Model (VLM) для автоматического распознавания и описания элементов одежды на изображении. Проект поддерживает работу в Jupyter Notebook для тестирования модели и предоставляет интерфейс в виде Telegram-бота для получения описаний одежды из изображений, отправленных в мессенджер.

## Основные возможности

- **Извлечение элементов одежды**: Модель анализирует изображение и возвращает текстовое описание одежды, включая такие характеристики, как цвет, стиль и узор.
- **Интерфейс Jupyter Notebook**: Возможность работы с изображениями для тестирования модели и вывода описаний прямо в ноутбуке.
- **Telegram-бот**: Telegram-бот для взаимодействия с моделью. Бот принимает изображения, анализирует их и возвращает пользователю описание каждого элемента одежды.

## Установка

1. **Клонируйте репозиторий**:

    ```bash
    git clone https://github.com/yourusername/clothing-description-extractor.git
    cd clothing-description-extractor
    ```

2. **Установите зависимости**:

    ```bash
    pip install transformers torch pillow requests python-telegram-bot
    ```

## Настройка и запуск

### Шаг 1: Запуск анализа изображений в Jupyter Notebook

- Откройте файл `Clothing_Description_Extractor.ipynb` в Jupyter Notebook.
- Укажите URL изображения для анализа в переменной `image_url`.
- Выполните ячейки, чтобы загрузить модель, проанализировать изображение и вывести описания одежды.

### Шаг 2: Настройка и запуск Telegram-бота

- Получите токен для Telegram-бота через [BotFather](https://t.me/BotFather) и замените значение `YOUR_TELEGRAM_BOT_TOKEN` на свой токен в коде.
- Запустите скрипт `clothing_description_extractor.py`, чтобы активировать бота.

    ```bash
    python clothing_description_extractor.py
    ```

- Отправьте изображение с одеждой боту в Telegram, и он ответит детализированным описанием элементов одежды на английском языке.

## Пример использования

- **Jupyter Notebook**: Позволяет провести анализ изображения и просмотреть результат в интерактивной среде.
- **Telegram-бот**: Получите описание элементов одежды, отправив боту изображение, где он выделит каждую вещь и укажет её основные визуальные характеристики.

## Зависимости

- `transformers` — для загрузки и использования Vision-Language модели.
- `torch` — для работы с моделью PyTorch.
- `pillow` — для обработки изображений.
- `requests` — для загрузки изображений по URL.
- `python-telegram-bot` — для взаимодействия с Telegram API.

## Лицензия

Проект распространяется под лицензией MIT. Подробнее см. в файле [LICENSE](LICENSE).

---
