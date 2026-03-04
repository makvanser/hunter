# Project Control & Skill Integration

## Global Skills Activation

Ты обязан использовать внешние инструкции, подключенные через "Skill Custom Paths". Для этого проекта активны следующие модули:

- **Loki Mode**: Используй все *.md файлы из директории `LokiModeRepo` для реализации цикла RARV (Reason, Act, Reflect, Verify).
- **Ralph Loop**: Используй инструкции из `RalphRepo` для обеспечения рекурсивной проверки до полной готовности (Definition of Done).
- **Engineering Core**: Применяй лучшие практики из `ClaudeBestPractices` (Борис Черни) для управления воркфлоу.

## Local Context Override

При наличии конфликтов локальные файлы проекта имеют приоритет над глобальными:

1. **Orchestration**: См. `@.agents/rules/orchestration.md` для логики планирования.
2. **Quality Standards**: См. `@.agents/rules/quality.md` для критериев приемки кода.
3. **Active Task Tracking**: Весь прогресс фиксируется строго в `@tasks/todo.md`.
4. **Knowledge Accumulation**: Ошибки и паттерны исправлений записываются в `@tasks/lessons.md`.

## Execution Protocol

1. **Bootstrap**: В начале каждой сессии проиндексируй глобальные репозитории и сопоставь их с текущим планом в `tasks/todo.md`.
2. **Thinking**: Перед любым действием используй тег <thinking>. Обоснуй выбор конкретного метода из Loki или Ralph.
3. **Subagent Offloading**: Для глубокого изучения множества .md файлов в глобальных репозиториях запускай субагента-исследователя. Не перегружай основной контекст сырым текстом документации.
4. **State Persistence**: Если задача прерывается, обнови статус в `tasks/todo.md`, чтобы при следующем запуске агент считал состояние из этого файла.

## Project Specifics

- Назначение: [Укажите кратко: Торговый бот V10 / Архитектурный скрипт Rhino]
- Стек: [Укажите технологии]
