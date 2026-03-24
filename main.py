"""
TimingGame — FastAPI бэкенд
Кадры из видео → Gemini Vision → реальный анализ событий.

Запуск:
    pip install -r requirements.txt
    python main.py
"""

import os
import json
import cv2
import base64
import asyncio
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings("ignore", category=FutureWarning, module="google")
import google.generativeai as genai

# ─── Конфигурация ───────────────────────────────────────────
VIDEO_FILE = "video.mp4"

# Рабочие модели (март 2026)
ALL_MODELS = [
    "gemini-2.5-flash",          # 10 RPM / 500 RPD free
    "gemini-2.5-flash-lite",     # 15 RPM / 1000 RPD free
    "gemini-2.5-pro",            # 5 RPM free
    "gemini-3-flash-preview",    # платный
    "gemini-3.1-pro-preview",    # платный
]

app = FastAPI(title="TimingGame Backend")

state = {
    "api_key": None,
    "models": [],
    "current_model": None,
    "configured": False,
}


# ══════════════════════════════════════════════════════════════
# GEMINI
# ══════════════════════════════════════════════════════════════
def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    state["api_key"] = api_key
    state["models"] = list(ALL_MODELS)
    state["current_model"] = ALL_MODELS[0]
    state["configured"] = True
    print(f"✓ Gemini → {ALL_MODELS[0]}")


async def call_gemini(contents, max_retries: int = 2):
    """Вызов с retry + fallback по цепочке моделей."""
    models_to_try = list(state["models"])
    last_error = None

    for model_name in models_to_try:
        print(f"  → {model_name}...")
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(contents)
                if not response.text or not response.text.strip():
                    print(f"  ✗ {model_name}: пустой ответ")
                    break
                state["current_model"] = model_name
                print(f"  ✓ {model_name} OK")
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                last_error = f"{model_name}: {err_str[:150]}"
                lo = err_str.lower()

                if any(x in lo for x in ["429", "quota", "rate limit", "resource has been exhausted"]):
                    print(f"  ⚠ 429 {model_name} ({attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep((attempt + 1) * 8)
                        continue
                    break  # следующая модель

                if "403" in err_str and "api key" in lo:
                    raise  # ключ битый

                if "404" in err_str or "not found" in lo:
                    print(f"  ✗ {model_name}: 404, пропускаю")
                    break

                if "400" in err_str:
                    print(f"  ✗ {model_name}: 400, пропускаю")
                    break

                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                    continue
                break

    raise RuntimeError(f"Все модели недоступны: {last_error}")


# ══════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ══════════════════════════════════════════════════════════════
class AnalyzeRequest(BaseModel):
    timestamp: float
    sport: str

class InitRequest(BaseModel):
    api_key: str


def get_video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0


def extract_frames(video_path: str, center_time: float,
                   num_frames: int = 5, step_sec: float = 0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Видео не найдено")
    duration = get_video_duration(video_path)
    result = []
    start = max(0, center_time - (num_frames // 2) * step_sec)
    for i in range(num_frames):
        t = start + i * step_sec
        if t > duration:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            result.append({"b64": base64.b64encode(buf).decode('utf-8'), "time": round(t, 2)})
    cap.release()
    return result


SPORT_EVENTS = {
    "football":   "гол, удар по воротам, фол, офсайд, угловой, пенальти, сейв",
    "esports":    "килл, хедшот, постановка бомбы, клатч, эйс, взрыв, перестрелка",
    "basketball": "бросок, трёхочковый, данк, блок-шот, перехват, фол",
    "boxing":     "удар, комбинация, нокдаун, уклонение, клинч, апперкот",
}


def build_prompt(sport: str, num_frames: int, center_idx: int) -> str:
    events = SPORT_EVENTS.get(sport, "ключевое спортивное событие")
    return f"""Ты — AI-судья для спортивной/игровой трансляции ({sport}).

Тебе даны {num_frames} последовательных кадров из видео (интервал ~0.5 секунды).
Игрок нажал кнопку ровно на кадре #{center_idx + 1}, считая что в этот момент
произошло ключевое событие.

Возможные события для {sport}: {events}.

Проанализируй ДИНАМИКУ: что происходит до, в момент нажатия, и после.
Оцени, действительно ли на экране происходит значимое событие.

ВЕРНИ СТРОГО JSON (без маркдауна, без ```, без лишнего текста):
{{
  "is_event": true или false,
  "confidence": число от 0 до 100,
  "event_name": "название события на русском (2-4 слова)",
  "comment": "дерзкий комментарий комментатора на русском (1-2 предложения)"
}}"""


def _err(name, comment, code):
    return {"is_event": False, "confidence": 0,
            "event_name": name, "comment": comment, "error": code}


# ══════════════════════════════════════════════════════════════
# РОУТЫ
# ══════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("index.html")

@app.get("/video.mp4")
async def serve_video():
    if not os.path.exists(VIDEO_FILE):
        raise HTTPException(404, "video.mp4 не найден")
    return FileResponse(VIDEO_FILE, media_type="video/mp4")

@app.get("/api/video-info")
async def video_info():
    if not os.path.exists(VIDEO_FILE):
        return {"exists": False, "duration": 0}
    return {"exists": True, "duration": round(get_video_duration(VIDEO_FILE), 2)}

@app.post("/api/init")
async def init_api(req: InitRequest):
    try:
        configure_gemini(req.api_key)
        return {"ok": True, "model": state["current_model"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/analyze")
async def analyze_moment(req: AnalyzeRequest):
    if not state["configured"]:
        return _err("Ошибка", "Gemini не подключён.", "not_configured")
    if not os.path.exists(VIDEO_FILE):
        return _err("Ошибка", "video.mp4 не найден.", "no_video")

    print(f"\n🔍 t={req.timestamp:.2f}с ({req.sport})")

    try:
        frames = extract_frames(VIDEO_FILE, req.timestamp)
    except Exception as e:
        return _err("Ошибка", f"Кадры: {e}", "frame_error")
    if not frames:
        return _err("Ошибка", "Кадры пустые.", "empty_frames")

    center_idx = len(frames) // 2
    prompt = build_prompt(req.sport, len(frames), center_idx)
    contents = [prompt]
    for i, f in enumerate(frames):
        tag = " ◀ КЛИК" if i == center_idx else ""
        contents.append(f"Кадр {i+1} (t={f['time']}с){tag}:")
        contents.append({"mime_type": "image/jpeg", "data": f["b64"]})

    try:
        raw = await call_gemini(contents)
        raw = raw.replace("```json", "").replace("```", "").strip()
        js, je = raw.find("{"), raw.rfind("}") + 1
        if js >= 0 and je > js:
            raw = raw[js:je]
        data = json.loads(raw)
        return {
            "is_event": bool(data.get("is_event", False)),
            "confidence": min(100, max(0, int(data.get("confidence", 0)))),
            "event_name": str(data.get("event_name", "Неизвестно"))[:50],
            "comment": str(data.get("comment", ""))[:200],
            "model_used": state["current_model"],
        }
    except json.JSONDecodeError:
        return _err("Ошибка", "ИИ вернул нечитаемый ответ.", "json_error")
    except RuntimeError as e:
        msg = str(e)
        if any(x in msg.lower() for x in ["429", "quota", "rate"]):
            return _err("Лимит API",
                        f"Квота исчерпана. Подожди ~60с или смени ключ. ({state['current_model']})",
                        "rate_limit")
        return _err("Ошибка AI", msg[:120], "all_failed")
    except Exception as e:
        return _err("Ошибка", str(e)[:120], "unknown")

@app.post("/api/summary")
async def game_summary(request: dict):
    if not state["configured"]:
        return {"summary": "ИИ не подключён."}

    history = request.get("history", [])
    name = request.get("name", "Игрок")
    sport = request.get("sport", "спорт")
    total = request.get("total_score", 0)
    mx = len(history) * 100

    lines = [f"R{h['round']}: {h['event']} — {'hit' if h['hit'] else 'miss'}, {h['pts']}pts"
             for h in history]
    prompt = f"""Ты AI-тренер. Игрок "{name}", {sport}.
{chr(10).join(lines)}
Итого: {total}/{mx}.
2-3 предложения: разбор + совет. Русский. Без маркдауна."""
    try:
        text = await call_gemini([prompt])
        return {"summary": text.replace("*", "").replace("#", "")}
    except:
        return {"summary": f"Итог: {total}/{mx}. {'Отличная игра!' if total > mx * 0.5 else 'Есть куда расти!'}"}


if __name__ == "__main__":
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        configure_gemini(env_key)

    print("=" * 55)
    print("⚡ TimingGame Backend — Gemini Vision")
    print(f"  http://127.0.0.1:8000")
    print(f"  Модели: {' → '.join(ALL_MODELS)}")
    print(f"  Видео: {VIDEO_FILE}")
    print("=" * 55)
    uvicorn.run(app, host="127.0.0.1", port=8000)
