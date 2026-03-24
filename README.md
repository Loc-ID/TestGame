# ⚡ TimingGame — Vision AI

Игрок смотрит спортивное/киберспортивное видео, жмёт кнопку в момент ключевого события — нейросеть анализирует кадры и выносит вердикт.

---

## Как это работает

```
Игрок жмёт "МОМЕНТ!"
       │
       ▼
Frontend фиксирует таймкод видео → POST /api/analyze
       │
       ▼
Python (OpenCV) вырезает 5 кадров вокруг момента клика (±1с)
       │
       ▼
Кадры → Gemini Vision API (мультимодальный запрос)
       │
       ▼
Gemini анализирует динамику и возвращает:
  { is_event: true, confidence: 98, event_name: "Гол!", comment: "..." }
       │
       ▼
Frontend показывает результат, очки, комментарий AI
```

---

## Запуск

**1. Клонируй репо**
```bash
git clone https://github.com/Loc-ID/TestGame.git
cd TestGame
```

**2. Установи зависимости**
```bash
pip install -r requirements.txt
```

**3. Положи видео**

Любой клип 20–60 секунд (футбол, CS2, бокс — что угодно). Назови `video.mp4`, положи в корень проекта рядом с `main.py`.

**4. Запусти**
```bash
python main.py
```

**5. Открой** [http://127.0.0.1:8000](http://127.0.0.1:8000)

Введи никнейм, выбери спорт, вставь Gemini API ключ → играй.

---

## Gemini API ключ

Бесплатный: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

Бесплатный тариф: 10 запросов/мин, 500 запросов/день на gemini-2.5-flash.
При исчерпании квоты система автоматически переключается на другую модель.

---

## Стек

| Компонент | Технология |
|-----------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Frame extraction | OpenCV |
| AI | Gemini Vision API (мультимодальный анализ) |
| Frontend | Vanilla JS, HTML/CSS |

---

## Структура

```
├── main.py            # FastAPI сервер: роуты, извлечение кадров, Gemini API
├── index.html         # Фронтенд: видеоплеер, кнопка, UI игры
├── video.mp4          # Видео для анализа (добавь сам)
└── requirements.txt   # Python-зависимости
```
