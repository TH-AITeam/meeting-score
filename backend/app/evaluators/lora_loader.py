"""vLLM への組織別 LoRA アダプタ動的ロード (Issue #83)。

vLLM を ``--enable-lora`` で起動すると、OpenAI 互換 API の拡張
``POST {base}/load_lora_adapter`` で起動後にアダプタを動的登録できる。

registry.yaml がサーバ起動後に更新された場合（再学習バッチでの hot-reload）、
起動時の ``--lora-modules`` には新アダプタが含まれないため、``model=<org_id>`` で
いきなり推論すると失敗する。そこで **アダプタを選択する前にロード**し、
ロードできたものだけを使う（できなければベースモデルに降格）。

## ロード済みキャッシュの挙動（重要）

成功した (endpoint, name, path) は **TTL 付き**でプロセス内にキャッシュし、毎リクエストの
重複ロードを避ける。バージョン更新で path が変われば別キーになり再ロードされる。

ただしこのキャッシュは vLLM 側の ``max-loras`` LRU 退避を**即時には追従できない**:
TTL 内にアダプタが LRU で押し出された場合、キャッシュは「ロード済み」と見なすため、
次の 1 リクエストはアダプタ未ロードのまま実行され失敗 → ベースモデルに fallback する
(遅延無効化 = lazy invalidation)。致命的ではない（必ずベースで応答は返る）が、
- TTL 経過後は再ロードを試みて自然回復する、
- 失敗を検知した呼び出し側は :func:`invalidate` で即座にキャッシュを落とせる、
という二段で緩和する。完全な追従が必要なら推論失敗フックからの ``invalidate`` 連携か
``max-loras`` を実運用組織数以上に取る運用を検討する。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Callable
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

# ロード済み (endpoint, lora_name, lora_path) -> 最終ロード時刻 (monotonic 秒)
_loaded: dict[tuple[str, str, str], float] = {}
# キー単位のロック（同一アダプタへの並行初回リクエストで重複 POST を防ぐ）
_key_locks: dict[tuple[str, str, str], threading.Lock] = {}
_key_locks_guard = threading.Lock()

DEFAULT_TTL_SECONDS = 300.0  # この秒数を超えたら再ロードを試みる（LRU 退避の遅延回復）

# (url, payload, timeout) -> (status_code, body)。テストで差し替え可能。
Poster = Callable[[str, dict, float], "tuple[int, str]"]


def _default_post(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    """vLLM の load_lora_adapter に POST し (status, body) を返す。

    urllib は 4xx/5xx で HTTPError を投げるが、vLLM は「既にロード済み」を
    4xx + body で返すことがある。HTTPError は status/body に変換して呼び出し側へ渡し、
    "already" 判定に流せるようにする（接続エラー等はそのまま送出）。
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "ignore")
    except HTTPError as e:  # 4xx/5xx は body を読んで status とともに返す
        body = e.read().decode("utf-8", "ignore") if e.fp is not None else ""
        return e.code, body


def _get_key_lock(key: tuple[str, str, str]) -> threading.Lock:
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def _is_fresh(key: tuple[str, str, str], ttl: float) -> bool:
    loaded_at = _loaded.get(key)
    return loaded_at is not None and (time.monotonic() - loaded_at) < ttl


def ensure_lora_loaded(
    endpoint: str,
    lora_name: str,
    lora_path: str | None,
    *,
    timeout: float = 10.0,
    ttl: float = DEFAULT_TTL_SECONDS,
    post: Poster | None = None,
) -> bool:
    """vLLM にアダプタがロード済みであることを保証する。成功で True。

    TTL 内にロード済みならキャッシュヒットで即 True。さもなくば
    ``load_lora_adapter`` に POST する。既ロード ("already" 応答) も True。
    失敗時は False を返し、呼び出し側はベースモデルに降格する。

    同期関数。async ハンドラからは ``asyncio.to_thread`` 等でイベントループを
    ブロックしないように呼ぶこと（urllib は最大 timeout 秒ブロックする）。
    """
    if not lora_path:
        return False
    key = (endpoint, lora_name, lora_path)
    if _is_fresh(key, ttl):  # ロック前の高速パス
        return True

    post = post or _default_post
    url = f"{endpoint.rstrip('/')}/load_lora_adapter"
    # キー単位ロックで同一アダプタへの並行 POST を 1 回に直列化する。
    with _get_key_lock(key):
        if _is_fresh(key, ttl):  # ロック取得中に他スレッドがロード済みなら再利用
            return True
        try:
            status, body = post(url, {"lora_name": lora_name, "lora_path": lora_path}, timeout)
        except Exception as e:  # 接続不可・タイムアウト等
            logger.warning("LoRA ロード呼び出しに失敗しました (%s): %s", lora_name, e)
            return False

        if 200 <= status < 300 or "already" in body.lower():
            _loaded[key] = time.monotonic()
            return True
        logger.warning(
            "LoRA ロードに失敗しました (%s): status=%s body=%s", lora_name, status, body[:200]
        )
        return False


def invalidate(endpoint: str, lora_name: str, lora_path: str) -> None:
    """ロード済みキャッシュから 1 エントリを落とす（次回リクエストで再ロードさせる）。

    推論時にアダプタ未ロードを検知した呼び出し側が呼ぶことで、LRU 退避に即追従できる。
    """
    _loaded.pop((endpoint, lora_name, lora_path), None)


def reset_loaded_cache() -> None:
    """ロード済みキャッシュをクリアする（主にテスト用）。"""
    _loaded.clear()


__all__ = ["ensure_lora_loaded", "invalidate", "reset_loaded_cache"]
