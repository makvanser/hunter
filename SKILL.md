---
name: Hunter Trading Bot (V23)
description: Institutional-grade crypto trading bot optimized for small capital ($100), featuring ML filtering, limit maker execution, and Order Book Imbalance (OBI) microstructure gating.
---

# Hunter Trading Bot (V23)

## 📌 Архитектура и Структура
Hunter — это асинхронный (aiohttp/asyncio) крипто-бот для Binance Futures. Система спроектирована по принципам институционального квант-трейдинга (Mid-Frequency) и оптимизирована для работы с депозитами от $100.

### Главные особенности V23:
1. **Maker-ордера (Комиссионный арбитраж):** Вместо `MARKET` ордеров используются строгие `LIMIT` ордера (`timeInForce: GTX` - Post Only) для уменьшения Taker комиссии (0.04%). Бот позиционирует себя на Best Bid / Best Ask с авто-таймаутом (30с).
2. **Order Book Imbalance (Микроструктура):** Фоновый WS-поток анализирует `bookTicker` (стакан) перед исполнением сделки. Если возникает огромная стена Ask/Bid, сделка блокируется (защита от Toxic Flow).
3. **ML-фильтрация сделок (Gradient Boosting):** Сборник из 14 фич (включая RSI, ATR, VWAP, Funding Rate, Open Interest) передается в предварительно обученную ML-модель. Сделка совершается только при уверенности (P(profit) > 60%).
4. **Умный риск-менеджмент:** Kelly criterion для сайзинга, глобальный `MAX_DRAWDOWN_PCT`, Cooldown (Breaker) на рынке после серии убытков (3 лосса) и фильтр флэта (ADX-regime).

### Основные компоненты (Файлы):
- `main.py`: Ядро приложения. Асинхронный WSS-цикл агрегации данных и принятия решений (Execution Gate).
- `provider.py` (`BinanceProvider`): Изолированный модуль получения данных. Поддерживает WS `kline` и `bookTicker`, а также REST API для исторической даты, Funding'а и Open Interest.
- `analysis.py`: Векторная математика, расчет индикаторов (RSI, Divergences, MACD, BB, ATR, VWAP). Выдает нормализованный `MarketState`.
- `execution.py` / `live_execution.py`: Ядро бумажного и реального трейдинга. Хранит IN-MEMORY позиции, трейлинги, рассчитывает PnL. `LiveTrader` отвечает за подписание API запросов (HMAC) и выставление реальных ордеров.
- `ml.py`: AI-фильтр на базе `sklearn.GradientBoostingClassifier`. Упаковка вектора состояния и прогноз вероятности прибыльного трейда.
- `database.py`: Логгер и база SQLite (для истории торгов и подсчета просадок).
- `config.py`: Единая точка управления всеми гиперпараметрами стратегии.

---

## 🚀 Операционное Управление и Запуск

### 1. Подготовка окружения
Установка всех зависимостей (особенно важен ML-стек):
```bash
pip install -r requirements.txt
# или: pip install aiohttp python-dotenv scikit-learn numpy 
```
В рабочей директории должен присутствовать файл `.env` с ключами:
```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
LIVE_TRADING=True     # или False для Paper Trading
USE_TESTNET=True      # или False для Binance Mainnet
```

### 2. Обучение ML-модели (Обязательный шаг!)
До первого запуска бота, сервер должен скачать исторические данные и подготовить ML-фильтр.
```bash
python backtest.py --train-ml
```
*Результат:* Бот произведет анализ истории, обучит нейросеть (скорость и точность) и сохранит веса в `ml_model.pkl`. Без этого файла бот будет пропускать сделки в слепую!

### 3. Запуск основного цикла
Бот спроетирован для фоновой работы (24/7). Используйте `tmux`, `screen` или `nohup`:
```bash
tmux new -s hunter
python main.py
```
*(Для выхода из `tmux` без прерывания бота нажать `Ctrl+B`, затем `D`)*.

---

## 🛠 Паттерны и Правила (Для Агентов)

При модификации системы Hunter AI агенты **обязаны** следовать следующим правилам:
- **Никаких `MARKET` ордеров!** Капитал мал. Исполнение на реальной бирже допускается только через методы `LiveTrader.open_limit_order` (Maker fee).
- **Single Responsibility Principle:** `provider.py` ничего не знает о трейдинге, только про HTTP/WSS. `analysis.py` ничего не знает о заявках, только математика. Формируйте логику в строгих границах модулей!
- **Zero-Trust к Балансу:** Любые расчеты риска (просадки) должны опираться на `LiveTrader.sync_balance()`, которая забирает чистые данные с биржи, а не на локальные переменные (чтобы исключить ошибки рассогласования).
- **Drawdown Protection:** Не убирай и не ослабляй Circuit Breakers (предохранители). Серии убыточных сделок убивают мелкие депозиты. `MAX_CONSECUTIVE_LOSSES` и `MAX_DRAWDOWN_PCT` трогать запрещено без веской институциональной математической модели.
- **Microstructure First:** Перед входом мы всегда чекаем OBI через `provider.get_bbo()`. Если стакан агрессивно против входа, логика должна возвращать сигнал в состояние `HOLD`.

## 📜 Чейнджлог версий (Исторический контекст)
- `V17`: Асинхронная архитектура и композитные сигналы (RSI Divs + MACD).
- `V22`: Внедрен ML-Gate (Градиентный бустинг по 12 фичам) и Kelly Criterion.
- `V23`: Переход на Maker-(Limit)-состояние, WSS bookTicker Imbalance, добавлены Institutional фичи (Funding и Open Interest) на 14-D пространство для ML-модели.
