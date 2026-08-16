import os
import traceback
from pathlib import Path

import anthropic


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env(Path(__file__).parent / ".env")

try:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "Reply with just the word OK"}],
    )
    print(response.content[0].text)
except Exception:
    traceback.print_exc()
