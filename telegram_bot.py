import os
import pickle
import telebot
import numpy as np
import pandas as pd
from lightfm import LightFM
from dotenv import load_dotenv
from telebot.types import ReplyKeyboardMarkup, KeyboardButton

# Загрузка модели рекомендательной системы
with open('model/FPLS_lightfm_model.pickle', 'rb') as f:
    model_data = pickle.load(f)

# Извлечение компонентов модели рекомендательной системы
lightfm_model = model_data['model']
df_lightfm = model_data['dataset']
item_features = model_data['item_features']
df_merged = model_data['df_merged']
df_matrix = model_data['df_matrix']

# Функция для получения рекомендаций
def recommendations_goods_for_stores(store_name, product_category, number_recommendations):
    # Проверка на наличие торговой точки в датасете
    if store_name not in df_matrix.columns:
        return (f'Торговая точка - {store_name} не найдена')
    # Извлечение всех товаров из выбранной категории
    candidate_products = df_merged[(df_merged['product_category'] == product_category)]
    # Проверка на наличие товаров по выбранной категории и торговой точке
    if candidate_products.empty:
        return (f'Нет данных о закупках в категории - {product_category} для торговой точки - {store_name}')  
    # Лист со списком наименований товаров, которые будут рекомендоваться
    product_names = candidate_products['product_name'].values
    try:
        # Определение идентификатора магазина для модели
        store_id = df_lightfm.mapping()[0][store_name]
        # Преобразование наименований товаров в идентификаторы
        product_id = [df_lightfm.mapping()[2][p] for p in product_names if p in df_lightfm.mapping()[2]]
        # Генерация рекомендаций
        generating_recommendations = lightfm_model.predict(store_id, product_id, item_features = item_features)
        # Сортировка товаров по убыванию предсказанных оценок и выбор топ-n товаров
        ratings_recommendations = np.argsort(-generating_recommendations)[:number_recommendations]
        top_product_names = [product_names[i] for i in ratings_recommendations]
        
        # Возвращение сгенерированных рекомендаций товаров для торговой точки
        return df_merged[df_merged['product_name'].isin(top_product_names)][['product_name', 'product_category', 'abc_classification', 'total_realization']]
    except KeyError as e:
        return (f'Ошибка: {str(e)}')

# Список доступных торговых точек
store_names = list(df_matrix.columns)
# Список доступных категорий товаров
product_categories = df_merged['product_category'].unique()

# Считывание токена из файла для инициализации бота
try:
    with open('telegram_bot_key.txt', 'r') as file:
        tg_token = file.read().strip()
    if not tg_token:
        raise ValueError('Файл пустой')
    tg_bot = telebot.TeleBot(tg_token)
except FileNotFoundError:
    print('Файл не найдей')
except ValueError as ve:
    print(f'Ошибка: {ve}')
except Exception as e:
    print(f'Неизвестная ошибка: {e}')

# Переменная для выбранной категории товара
selected_product_category = None
# Переменная для выбранной торговой точки
selected_store_name = None
# Переменная для выбранного количетсва рекомендаций
selected_number_recommendations = 0

# Функция вывода главного меню бота
def show_main_menu(message):
    markup = ReplyKeyboardMarkup(resize_keyboard = True)
    markup.add(KeyboardButton('🎯 Получить рекомендации'))
    tg_bot.send_message(message.chat.id, '🚀 Нажмите на кнопку, чтобы начать заполнение данных для получения рекомендаций', reply_markup = markup)

# Функция для начала работы бота
@tg_bot.message_handler(commands = ['start'])
def start(message):
    markup = ReplyKeyboardMarkup(resize_keyboard = True)
    # Добавление кнопки 'Получить рекомендации'
    markup.add(KeyboardButton('🎯 Получить рекомендации'))
    # Вывод текста перед кнопкой
    tg_bot.send_message(message.chat.id,'🚀 Нажмите на кнопку, чтобы начать заполнение данных для получения рекомендаций', reply_markup = markup)

# Функция для выбора категории товара
@tg_bot.message_handler(func = lambda message: message.text == '🎯 Получить рекомендации')
def product_category_selection(message):
    # Создание клавиатуры с кнопками, где одна кнопка - одна категория товара
    markup = ReplyKeyboardMarkup(resize_keyboard = True, one_time_keyboard = True)
    # Добавление кнопки для каждой категории товара
    for product_category in product_categories:
        markup.add(KeyboardButton(product_category))
    # Вывод текста перед кнопками
    tg_bot.send_message(message.chat.id, '🏷 Выберите категорию товара', reply_markup = markup)

# Функция для сохранения выбранной категории товара
@tg_bot.message_handler(func = lambda message: message.text in product_categories)
def saved_product_category(message):
    # Использование глобальной переменной для хранения выбранной категории товара
    global selected_product_category
    # Получение выбранной категории товара
    selected_product_category = message.text
    # Вызов функции для выбора торговой точки
    store_name_selection(message)

# Функция для выбора торговой точки
def store_name_selection(message):
     # Создание клавиатуры с кнопками, где одна кнопка - одна торговая точка
    markup = ReplyKeyboardMarkup(resize_keyboard = True, one_time_keyboard = True)
     # Добавление кнопки для каждой торговой точки
    for store_name in store_names:
        markup.add(KeyboardButton(store_name))
    # Вывод текста перед кнопками
    tg_bot.send_message(message.chat.id, '🏪 Выберите торговую точку', reply_markup = markup)

# Функция для сохранения выбранной торговой точки
@tg_bot.message_handler(func=lambda message: message.text in store_names)
def saved_store_name(message):
    # Использование глобальной переменной для хранения выбранной торговой точки
    global selected_store_name
    # Получение выбранной торговой точки
    selected_store_name = message.text
    # Вызов функции для выбора количетсва рекомендаций
    number_recommendations_selection(message)

# Функция для выбора количества рекомендаций
def number_recommendations_selection(message):
    # Создание клавиатуры с кнопками, где одна кнопка - колиество рекомендаций
    markup = ReplyKeyboardMarkup(resize_keyboard = True, one_time_keyboard = True)
    # Создание первого ряда с количеством рекомендаций
    markup.row(KeyboardButton('3'), KeyboardButton('6'))
    # Создание второго ряда с количеством рекомендаций
    markup.row(KeyboardButton('9'), KeyboardButton('12'))
    # Вывод текста перед кнопками
    tg_bot.send_message(message.chat.id, '🔢 Выберите количество рекомендаций', reply_markup = markup)

# Функция для сохранения выбранного количетсва рекомендаций
@tg_bot.message_handler(func = lambda message: message.text in ['3', '6', '9', '12'])
def saved_number_recommendations(message):
    # Использование глобальной переменной для хранения выбранного количетсва рекомендаций
    global selected_number_recommendations, selected_product_category, selected_store_name
    # Получение выбранного количетсва рекомендаций
    selected_number_recommendations = int(message.text)
    # Создание клавиатуры с кнопками, где одна кнопка - вызов функции генерации рекомендаций, вторая - начать заполнение сначала
    markup = ReplyKeyboardMarkup(resize_keyboard = True, row_width = 2)
    # Создание кнопок, где одна кнопка - вызов функции генерации рекомендаций, вторая - начать заполнение сначала
    markup.add(KeyboardButton('✅ Сгенерировать рекомендации'), KeyboardButton('🔄 Заполнить заново'))
    # Сообщение с информацией о введённых параметрах
    info = (
        "📋 <b>Выбранные параметры:</b>\n\n"
        f"  🏷 Категория товара: <i>{selected_product_category}</i>\n"
        f"  🏪 Торговая точка: <i>{selected_store_name}</i>\n"
        f"  🔢 Количество рекомендаций: <i>{selected_number_recommendations}</i>\n\n"
        "📌 Выберите действие:"
    )
    # Отправка информации о введённых параметрах
    tg_bot.send_message(message.chat.id, info, reply_markup = markup, parse_mode = 'HTML')

# Функция перезапуска выбора параметров для генерации рекомендаций
@tg_bot.message_handler(func = lambda message: message.text == '🔄 Заполнить заново')
def restart_process(message):
    product_category_selection(message)

# Функция для отправки рекомендаций
@tg_bot.message_handler(func = lambda message: message.text == '✅ Сгенерировать рекомендации')
def output_recommendations(message):
    try:
        # Получение рекомендаций из модели
        recommendations = recommendations_goods_for_stores(store_name = selected_store_name, product_category = selected_product_category, number_recommendations = selected_number_recommendations)
        # Если на выходе сообщение об ошибке, то выводим её
        if isinstance(recommendations, str):
            tg_bot.send_message(message.chat.id, recommendations)
        # Иначе, если датасет, то выводим его 
        else:
            # Вызов функции отвечающей за оформления вывода рекомендаций
            result = recommendation_output_format(recommendations)
            # Отправка оформленных рекомендаций
            tg_bot.send_message(message.chat.id, result, parse_mode = 'HTML')
    except Exception as e:
        # Вывод сообщения об исключении
        error_message = (f'Ошибка при генерации рекомендаций:\n{str(e)}')
        # Вывод сообщения об исключении
        tg_bot.send_message(message.chat.id, error_message)
    finally:
        show_main_menu(message)

#Функция оформления вывода рекомендаций
def recommendation_output_format(recommendations):
    count = 1
    # Заголовок сообщения с выводом рекомендаций
    message = "<b>🎯 Рекомендации по выбранным параметрам:</b>\n\n"
    # Вывод всех рекмоендованных товаров 
    for i, row in recommendations.iterrows():
        message += (
            f"<b>№ {count}</b> 🍎 Наименование товара - <i>{row['product_name']}</i>\n"
            f"  🏷 Категория товара - <i>{row['product_category']}</i>\n"
            f"  ⚖️ ABC-классификация - <i>{row['abc_classification']}</i>\n\n"
        )
        count += 1
    return message

# Функция для остановки бота
@tg_bot.message_handler(commands = ['stop'])
def stop_bot(message):
    tg_bot.send_message(message.chat.id, 'Бот остановлен')
    tg_bot.stop_polling()
    exit(0)

# Запуск бота
tg_bot.polling(none_stop = True)