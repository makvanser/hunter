# План Развития: Фаза 7 — Live Testnet Trading

## 1. Ключи и Базовая Конфигурация

- [x] `API_KEY` и `API_SECRET`.
- [x] Переключение `BASE_URL` на `https://testnet.binancefuture.com`.

## 2. `live_execution.py`

- [x] Класс `LiveTrader` с методами подписи HMAC-SHA256.
- [x] Открытие рыночных позиций (Market).
- [x] Наследование динамического трейлинга SL/TP из `PaperTrader`.
- [x] Мониторинг баланса через API.

## 3. Внедрение

- [x] Флаг `LIVE_TRADING` в `config.py`.
- [x] Замена `PaperTrader` на `LiveTrader` в `main.py` при включенном флаге.

## 4. QA

- [x] Получение баланса.
- [x] Открытие тестовой микро-сделки.
