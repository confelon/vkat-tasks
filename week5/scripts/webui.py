"""Минимальный web-интерфейс: строка ввода, ответ сервера и сравнение «до/после».

Без истории, без сторонних зависимостей: стандартный http.server отдаёт
страницу и обрабатывает два запроса.
  POST /chat    — вопрос к vLLM тем же клиентом и с теми же параметрами,
                  что scripts/client.py.
  POST /compare — тот же вопрос локально базовой модели и базе + адаптер,
                  как scripts/validate_inference.py с флагом --no-adapter и без.
Кнопка сравнения включается только когда обучение прошло и адаптер лежит
в output/adapter/.

    docker compose run --rm -p 8080:8080 app python scripts/webui.py
    # затем открыть http://localhost:8080
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from openai import OpenAI

from app.config import ProjectConfig
from app.inference import InferenceService

# Те же адрес и имя модели, что в scripts/client.py: страница живёт в
# контейнере, поэтому ходит к vLLM по имени сервиса в сети Compose.
BASE_URL = "http://vllm:8000/v1"
MODEL_NAME = "alpaca-lora"
PORT = 8080

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>alpaca-lora</title>
<style>
  body { font: 16px system-ui, sans-serif; max-width: 64rem; margin: 3rem auto; padding: 0 1rem; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; font: inherit; padding: .5rem; }
  button { font: inherit; padding: .5rem 1rem; }
  #hint { color: #888; }
  #answers { display: flex; gap: 1rem; }
  #answers > div { flex: 1; }
  pre { white-space: pre-wrap; background: #f4f4f4; padding: 1rem; min-height: 4rem; }
</style>
<h1>alpaca-lora</h1>
<form id="form">
  <input id="prompt" autofocus placeholder="Ask something" required>
  <button formaction="/chat">Send</button>
  <button id="compare" formaction="/compare">Compare base vs LoRA</button>
</form>
<p id="hint">Comparison needs a trained adapter in output/adapter/: run scripts/train.py first.</p>
<div id="answers"></div>
<script>
  const LABELS = {
    answer: "alpaca-lora (vLLM)",
    base: "base model",
    lora: "base model + LoRA adapter",
  };
  const adapterReady = ADAPTER_READY;
  const form = document.getElementById("form");
  const prompt = document.getElementById("prompt");
  const answers = document.getElementById("answers");
  document.getElementById("compare").disabled = !adapterReady;
  document.getElementById("hint").hidden = adapterReady;

  const show = (results) => {
    answers.replaceChildren();
    for (const [key, text] of Object.entries(results)) {
      const block = document.createElement("div");
      block.appendChild(document.createElement("h3")).textContent = LABELS[key];
      block.appendChild(document.createElement("pre")).textContent = text;
      answers.appendChild(block);
    }
  };

  form.onsubmit = async (event) => {
    event.preventDefault();
    for (const button of form.querySelectorAll("button")) button.disabled = true;
    show({[event.submitter.formAction.endsWith("/chat") ? "answer" : "lora"]: "..."});
    const response = await fetch(event.submitter.formAction, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt: prompt.value}),
    });
    show(await response.json());
    form.querySelector("button").disabled = false;
    document.getElementById("compare").disabled = !adapterReady;
  };
</script>
"""

client = OpenAI(base_url=BASE_URL, api_key="not-needed")
config = ProjectConfig()
# Сравнение «до/после» есть только после обучения: без адаптера кнопка
# выключена и модели на GPU не грузятся.
adapter_ready = (config.adapter_dir / "adapter_config.json").exists()
# Заполняется в main(): "base" — чистая модель, "lora" — она же с адаптером.
local: dict[str, InferenceService] = {}


class Handler(BaseHTTPRequestHandler):
    """GET — страница, POST /chat — ответ vLLM, POST /compare — «до/после»."""

    def do_GET(self) -> None:
        page = PAGE.replace("ADAPTER_READY", json.dumps(adapter_ready))
        self._reply("text/html; charset=utf-8", page.encode())

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        prompt = json.loads(self.rfile.read(length))["prompt"]
        if self.path == "/chat":
            results = {"answer": self._ask_vllm(prompt)}
        else:
            results = {name: service.generate(prompt) for name, service in local.items()}
        self._reply("application/json", json.dumps(results).encode())

    def _ask_vllm(self, prompt: str) -> str:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0,
            stop=["###"],
        )
        return response.choices[0].message.content

    def _reply(self, content_type: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    if adapter_ready:
        local["base"] = InferenceService(config, use_adapter=False)
        local["lora"] = InferenceService(config, use_adapter=True)
    print(f"Web UI on http://localhost:{PORT} (compare: {'on' if adapter_ready else 'off'})")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
