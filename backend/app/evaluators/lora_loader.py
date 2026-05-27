"""vLLM への組織別 LoRA アダプタ動的ロード (Issue #83)。

vLLM を ``--enable-lora`` で起動すると、OpenAI 互換 API の拡張
``POST {base}/load_lora_adapter`` で起動後にアダプタを動的登録できる。

registry.yaml がサーバ起動後に更新された場合（再学習バッチでの hot-reload）、
起動時の ``--lora-modules`` には新アダプタが含まれないため、``model=<org_id>`` で
いきなり推論すると失敗する。そこで **アダプタを選択する前にロード**し、
ロードできたものだけを使う（できなければベースモデルに降格）。

成功した (endpoint, name, path) はプロセス内でキャッシュし、毎リクエストの
重複ロードを避ける。バージョン更新で path が変われば再ロードされる。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ロード済み (endpoint, lora_name, lora_path) のキャッシュ
_loaded: set[tuple[str, str, str]] = set()

# (url, payload, timeout) -> (status_code, body)。テストで差し替え可能。
Poster = Callable[[str, dict, float], "tuple[int, str]"]


def _default_post(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "ignore")


def ensure_lora_loaded(
    endpoint: str,
    lora_name: str,
    lora_path: str | None,
    *,
    timeout: float = 10.0,
    post: Poster | None = None,
) -> bool:
    """vLLM にアダプタがロード済みであることを保証する。成功で True。

    既にロード済み (キャッシュ or サーバ側で "already" 応答) も True 扱い。
    失敗時は False を返し、呼び出し側はベースモデルに降格する。
    """
    if not lora_path:
        return False
    key = (endpoint, lora_name, lora_path)
    if key in _loaded:
        return True

    post = post or _default_post
    url = f"{endpoint.rstrip('/')}/load_lora_adapter"
    try:
        status, body = post(url, {"lora_name": lora_name, "lora_path": lora_path}, timeout)
    except Exception as e:
        logger.warning("LoRA ロード呼び出しに失敗しました (%s): %s", lora_name, e)
        return False

    if 200 <= status < 300 or "already" in body.lower():
        _loaded.add(key)
        return True
    logger.warning(
        "LoRA ロードに失敗しました (%s): status=%s body=%s", lora_name, status, body[:200]
    )
    return False


def reset_loaded_cache() -> None:
    """ロード済みキャッシュをクリアする（主にテスト用）。"""
    _loaded.clear()


__all__ = ["ensure_lora_loaded", "reset_loaded_cache"]
