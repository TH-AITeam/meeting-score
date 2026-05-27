"""組織別 LoRA アダプタの解決 (Issue #83)。

推論リクエストの ``org_id`` から、使うべき LLM モデル（組織別 LoRA アダプタ名 or
ベースモデル）とフォールバック種別を決める。アダプタは #82 (`scripts/retrain_loras.py`)
が `adapters/{org_id}/{version}/` に作り、`adapters/registry.yaml` に
``organizations: {org_id: {active_version: vN}}`` で登録する。

解決の 3 ケース (Epic #77):
  Case 1: アダプタあり        → そのアダプタで評価 (model=org_id)
  Case 2: 重みプロファイルあり → ベースモデル + 組織別重み (#81 の出力)
  Case 3: 両方なし            → ベースモデル + デフォルト重み

registry.yaml は mtime 監視でホットリロードする（再学習バッチが更新したら
プロセス再起動なしで反映）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# backend/app/evaluators/adapter_resolver.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "adapters" / "registry.yaml"
DEFAULT_ADAPTERS_ROOT = REPO_ROOT / "adapters"
DEFAULT_WEIGHTS_PROFILE_DIR = REPO_ROOT / "config" / "weights_profile"

KIND_ADAPTER = "adapter"
KIND_WEIGHTS_PROFILE = "weights_profile"
KIND_DEFAULT = "default"


@dataclass(frozen=True)
class AdapterChoice:
    """org_id に対する解決結果。"""

    org_id: str
    kind: str  # 'adapter' | 'weights_profile' | 'default'
    model: str  # LLM に投げるモデル名（アダプタ名 or ベースモデル）
    version: str | None = None
    adapter_path: str | None = None
    weights_profile_path: str | None = None


class AdapterResolver:
    """registry.yaml と重みプロファイルから org_id を解決する（mtime ホットリロード）。"""

    def __init__(
        self,
        base_model: str,
        registry_path: Path | str = DEFAULT_REGISTRY,
        adapters_root: Path | str = DEFAULT_ADAPTERS_ROOT,
        weights_profile_dir: Path | str = DEFAULT_WEIGHTS_PROFILE_DIR,
    ) -> None:
        self._base_model = base_model
        self._registry_path = Path(registry_path)
        self._adapters_root = Path(adapters_root)
        self._weights_profile_dir = Path(weights_profile_dir)
        self._registry: dict = {"organizations": {}}
        self._mtime: float | None = None

    def _load_registry(self) -> None:
        """registry を mtime 監視でロード/再ロードする。"""
        if not self._registry_path.exists():
            self._registry = {"organizations": {}}
            self._mtime = None
            return
        mtime = self._registry_path.stat().st_mtime
        if mtime == self._mtime:
            return
        try:
            data = yaml.safe_load(self._registry_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            logger.exception(
                "registry.yaml のパースに失敗。前回値を維持します: %s", self._registry_path
            )
            return
        if not isinstance(data, dict):
            data = {}
        data.setdefault("organizations", {})
        self._registry = data
        self._mtime = mtime

    def _safe_adapter_path(self, org_id: str, version: str) -> Path | None:
        """adapters_root 配下に収まる adapters/{org_id}/{version} を安全に解決する。"""
        try:
            root = self._adapters_root.resolve()
            candidate = root.joinpath(org_id, version).resolve()
            candidate.relative_to(root)  # traversal 防止
        except (ValueError, OSError):
            logger.warning("不正なアダプタパス: org_id=%s version=%s", org_id, version)
            return None
        return candidate

    def resolve(self, org_id: str) -> AdapterChoice:
        """org_id を AdapterChoice に解決する。"""
        self._load_registry()
        entry = self._registry.get("organizations", {}).get(org_id)
        version = entry.get("active_version") if isinstance(entry, dict) else None

        if version:
            adapter_path = self._safe_adapter_path(org_id, str(version))
            if adapter_path is not None and adapter_path.exists():
                return AdapterChoice(
                    org_id=org_id,
                    kind=KIND_ADAPTER,
                    model=org_id,  # vLLM の --lora-modules で org_id=path 登録される想定
                    version=str(version),
                    adapter_path=str(adapter_path),
                )
            # 登録はあるが実体が無い → 降格してフォールバック
            logger.warning(
                "アダプタが登録済みだが実体が見つかりません。フォールバックします: org_id=%s version=%s",
                org_id,
                version,
            )

        profile = self._weights_profile_dir / f"{org_id}.yaml"
        if profile.exists():
            return AdapterChoice(
                org_id=org_id,
                kind=KIND_WEIGHTS_PROFILE,
                model=self._base_model,
                weights_profile_path=str(profile),
            )

        return AdapterChoice(org_id=org_id, kind=KIND_DEFAULT, model=self._base_model)

    def lora_modules(self) -> list[str]:
        """vLLM 起動用の ``--lora-modules`` 引数 (``org_id=path``) を registry から作る。"""
        self._load_registry()
        out: list[str] = []
        for org_id, entry in sorted(self._registry.get("organizations", {}).items()):
            version = entry.get("active_version") if isinstance(entry, dict) else None
            if not version:
                continue
            path = self._safe_adapter_path(org_id, str(version))
            if path is not None and path.exists():
                out.append(f"{org_id}={path}")
        return out


__all__ = [
    "KIND_ADAPTER",
    "KIND_DEFAULT",
    "KIND_WEIGHTS_PROFILE",
    "AdapterChoice",
    "AdapterResolver",
]
