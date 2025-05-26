import requests

try:
    r = requests.get("http://localhost:11434")
    print("Ollama is running!" if r.status_code == 200 else "Ollama is not responding.")
except Exception as e:
    print("Ollama is not running:", e)

