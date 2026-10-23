from dotenv import load_dotenv   # функция, которая читает файл .env
from google import genai          # официальный SDK Gemini

load_dotenv()                     # переносит GEMINI_API_KEY из .env в переменные среды
client = genai.Client()           # клиент сам находит GEMINI_API_KEY, передавать ключ в код НЕ нужно

# Отправляем запрос модели
interaction = client.interactions.create(
    model="gemini-3.1-flash-lite",   # самая дешёвая модель
    input="Сколько комнат: 'Сдам 2к квартиру, 54 м2, Оболонь'? Ответь одним числом.",
)

print(interaction.output_text)    # текст ответа модели
print(interaction.usage)          # сколько токенов ушло: пригодится для расчёта стоимости