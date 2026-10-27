"""Запуск одной командой.

python main.py              — все прогоны (матрица модель × промпт × семплинг) + метрики
python main.py --metrics    — только пересчитать метрики по готовым results/*.jsonl
python main.py --quick      — только Gemini v2 t0 на всех текстах + метрики (≈4 мин)
"""
import subprocess
import sys

PY = sys.executable                     # тот же Python, что запустил main.py (из .venv)

# (провайдер, семплинг, промпт, какие тексты)
# Ollama на CPU ≈ 50–80 с/док → гоняем только golden set (10 текстов)
EXPERIMENTS = [
    ("gemini", "t0",  "v1", "all"),
    ("gemini", "t07", "v1", "all"),
    ("gemini", "t0",  "v2", "all"),
    ("gemini", "t07", "v2", "all"),
    ("ollama", "t0",  "v1", "golden"),
    ("ollama", "t07", "v1", "golden"),
    ("ollama", "t0",  "v2", "golden"),
    ("ollama", "t07", "v2", "golden"),
]


def run(args):
    print(">>", " ".join(args))
    subprocess.run([PY, *args], check=True)   # check=True: если шаг упал — останавливаемся


if __name__ == "__main__":
    flag = sys.argv[1] if len(sys.argv) > 1 else ""
    if flag == "--quick":
        run(["src/run.py", "gemini", "t0", "v2", "all"])
    elif flag != "--metrics":
        for exp in EXPERIMENTS:
            run(["src/run.py", *exp])
    run(["src/metrics.py"])
