MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "ollama": "qwen2.5:3b",
}


SAMPLING = {
    "t0":  {"temperature": 0.0, "top_p": 1.0},   
    "t07": {"temperature": 0.7, "top_p": 0.9},   
}

# Цена за 1 млн токенов, $ (платный уровень Gemini; локальная модель — 0)
PRICES = {
    "gemini": {"input": 0.25, "output": 1.50},
    "ollama": {"input": 0.0,  "output": 0.0},
}

TIMEOUT_S = {"gemini": 60, "ollama": 300} 
PAUSE_S = {"gemini": 5, "ollama": 0}        
MAX_RETRIES = 3                             

GOLDEN_IDS = ["001", "002", "008", "011", "013", "014", "017", "020", "023", "027"]