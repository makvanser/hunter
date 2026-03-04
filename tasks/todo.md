# V23 Institutional Upgrade

## Phase 1: Maker (Limit) Execution

- [ ] Описать `LIMIT_ORDER_TIMEOUT_SEC` параметров в `config.py`
- [ ] В `PaperTrader` (mock) добавить логику: симуляция limit исполнения по цене закрытия бара
- [ ] В `live_execution.py` переписать `execute_trade`:
  - Использовать `type='LIMIT'`, `timeInForce='GTC'`
  - Запрашивать bookTicker / текущую цену перед отправкой
  - Выставлять ордер. Если не заполнен за 30 сек (через API поллинг) -> отмена
- [ ] Протестировать выставление Limit ордеров на Binance Testnet

## Phase 2: Microstructure & Order Flow (BBO)

- [ ] Добавить подписку на `bookTicker` в `provider.py`
- [ ] Кэшировать Best Bid / Ask и объемы в `provider.py`
- [ ] В `main.py` добавить этап "Execution Gate": перед вызовом `execute_trade`, вычислить OBI (Order Book Imbalance).
- [ ] Применить фильтр: отменить сделку, если OBI против нас.

## Phase 3: Alternative Data for ML

- [ ] Найти эндпоинты Binance для Funding Rate и Open Interest
- [ ] Обновить сбор исторических данных `backtest.py`
- [ ] Добавить в `MarketState`
- [ ] Переобучить `ml_model.pkl`
