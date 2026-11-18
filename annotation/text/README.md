# Ручные отзывы

Источник - [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews), на странице указана CC0. Для разметки отобрано 120 отзывов длиной 5-80 слов; рейтинги исключены из заданий. Нормализованные дубликаты и совпадения с UCI проверены, смысловые повторы возможны.

`annotations_A1.json` и `annotations_A2.json` - исходные экспорты, `label_studio.xml` - интерфейс разметки. В `manual_text_agreed_112.json` сохранены 112 отзывов, где метки совпали: 55 positive и 57 negative.

В экспорте A2 присутствуют предсказания: у 118 ответов origin `prediction`, у двух - `prediction-changed`. По этим данным нельзя подтвердить независимость разметки или установить влияние предсказаний на ответы.
