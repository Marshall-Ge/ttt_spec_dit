# -*- coding: utf-8 -*-
"""M8: Version Registry — base model 版本隔离.

确保 LoRA adapter 不会跨 base model 版本静默复用。

核心规则:
  1. LoRA adapter 必须标记其训练的 base_model_version。
  2. 检测到 base model 版本变化时:
     - 现有 adapter 标记为 "stale" (不自动下线, 但禁止复用)
     - 新训练必须从零初始化开始 (不能 warm-start 自旧版本)
     - M2 OnlineCalibrator 不受影响 (统计量自然跟随新分布收敛)
  3. ``is_adapter_valid(adapter_ver, base_ver)`` 严格检查版本匹配。

版本标识来源:
  - 从 checkpoint 路径提取 (如 ``dit_2_256`` → ``dit-v1.0``)
  - 从模型 config 的 ``_name_or_path`` 字段
  - 人工指定的版本标签
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class AdapterStatus(Enum):
    ACTIVE = "active"          # 版本匹配, 可安全使用
    STALE = "stale"            # 版本不匹配, 需重新验证
    UNKNOWN = "unknown"        # 尚未关联版本信息


@dataclass
class AdapterRecord:
    """Single LoRA adapter 的版本记录."""
    path: str
    adapter_version: str
    base_model_version: str
    status: AdapterStatus = AdapterStatus.UNKNOWN
    created_at: str = ""
    last_validated_at: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class VersionEvent:
    """版本变更事件日志."""
    timestamp: str
    old_version: str
    new_version: str
    action: str  # "swap_detected", "adapter_staled", "calibrator_adapted", "rollback"
    details: str = ""


class VersionRegistry:
    """Base model version registry and adapter version isolation.

    Usage::

        registry = VersionRegistry(state_dir="./output/vfl_checkpoints")

        # On startup / model load:
        current_ver = registry.detect_version(model_path="/root/.../dit_2_256")
        registry.set_current_version(current_ver)

        # Before loading an adapter:
        if registry.is_adapter_valid(adapter_record):
            load_adapter(...)
        else:
            print("Adapter is stale — re-training required")

        # After base model update:
        registry.on_version_change("dit-v2.0")
    """

    def __init__(self, state_dir: str = "./output/vfl_checkpoints"):
        self.state_dir = state_dir
        self._current_version: str = "unknown"
        self._previous_versions: List[str] = []
        self._adapters: List[AdapterRecord] = []
        self._events: List[VersionEvent] = []

        os.makedirs(state_dir, exist_ok=True)
        self._state_path = os.path.join(state_dir, "version_registry.json")
        self._load_state()

    # ------------------------------------------------------------------
    # Version detection
    # ------------------------------------------------------------------

    def detect_version(self, model_path: str) -> str:
        """Detect base model version from model path.

        Heuristic:
          1. Extract directory name as version hint
          2. Hash the checkpoint file for a content-based fingerprint
          3. Combine into a compact version string

        Parameters
        ----------
        model_path : str
            Path to the model directory or checkpoint file.

        Returns
        -------
        version : str
            e.g. "dit_2_256-a1b2c3d4"
        """
        # Extract name from path
        norm = os.path.normpath(model_path)
        parts = [p for p in norm.split(os.sep) if p]

        # Find the most specific model name
        model_name = "unknown"
        for part in reversed(parts):
            if any(kw in part.lower() for kw in ["dit", "pixart", "model"]):
                model_name = part
                break

        # Try to hash the checkpoint for content fingerprint
        ckpt_hash = ""
        ckpt_path = os.path.join(model_path, "transformer",
                                 "diffusion_pytorch_model.bin")
        if not os.path.exists(ckpt_path):
            # Try direct .bin
            if os.path.isfile(model_path) and model_path.endswith(".bin"):
                ckpt_path = model_path

        if os.path.exists(ckpt_path):
            # Hash first 1MB of the checkpoint (fast + stable)
            with open(ckpt_path, "rb") as f:
                head = f.read(1_048_576)
            ckpt_hash = hashlib.md5(head).hexdigest()[:8]

        if ckpt_hash:
            return f"{model_name}-{ckpt_hash}"
        return model_name

    @staticmethod
    def detect_version_simple(model_path: str) -> str:
        """Simple version detection: just use the directory name."""
        norm = os.path.normpath(model_path)
        parts = [p for p in norm.split(os.sep) if p]
        for part in reversed(parts):
            if any(kw in part.lower() for kw in ["dit", "pixart", "model"]):
                return part
        return parts[-1] if parts else "unknown"

    # ------------------------------------------------------------------
    # Current version management
    # ------------------------------------------------------------------

    def set_current_version(self, version: str):
        """Set the current base model version."""
        if version != self._current_version and self._current_version != "unknown":
            self.on_version_change(version)
        else:
            self._current_version = version
            self._save_state()

    @property
    def current_version(self) -> str:
        return self._current_version

    # ------------------------------------------------------------------
    # Version change handling
    # ------------------------------------------------------------------

    def on_version_change(self, new_version: str):
        """Handle base model version change.

        Actions:
          1. Log the event
          2. Mark all adapters for the old version as STALE
          3. Record old version in history
          4. NOT: does NOT modify calibrator (that's handled by OnlineCalibrator)
        """
        old = self._current_version
        self._previous_versions.append(old)
        self._current_version = new_version

        # Log event
        self._events.append(VersionEvent(
            timestamp=datetime.now().isoformat(),
            old_version=old,
            new_version=new_version,
            action="swap_detected",
            details=f"Base model changed: {old} → {new_version}",
        ))

        # Stale all adapters
        for adapter in self._adapters:
            if adapter.status == AdapterStatus.ACTIVE:
                adapter.status = AdapterStatus.STALE
                self._events.append(VersionEvent(
                    timestamp=datetime.now().isoformat(),
                    old_version=old,
                    new_version=new_version,
                    action="adapter_staled",
                    details=f"Adapter {adapter.adapter_version} marked stale "
                            f"(was for {adapter.base_model_version})",
                ))

        self._save_state()

    # ------------------------------------------------------------------
    # Adapter validation
    # ------------------------------------------------------------------

    def register_adapter(self,
                         adapter_path: str,
                         adapter_version: str,
                         base_model_version: str,
                         metadata: Optional[Dict] = None) -> AdapterRecord:
        """Register a new LoRA adapter in the registry."""
        record = AdapterRecord(
            path=adapter_path,
            adapter_version=adapter_version,
            base_model_version=base_model_version,
            status=(AdapterStatus.ACTIVE
                    if base_model_version == self._current_version
                    else AdapterStatus.STALE),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._adapters.append(record)
        self._save_state()
        return record

    def is_adapter_valid(self, record: AdapterRecord) -> bool:
        """Check if an adapter is safe to use with the current base model.

        Returns True ONLY if:
          - adapter's base_model_version matches current version
          - adapter status is ACTIVE (not STALE/UNKNOWN)
        """
        if record.status != AdapterStatus.ACTIVE:
            return False
        if record.base_model_version != self._current_version:
            # Auto-mark as stale
            record.status = AdapterStatus.STALE
            self._save_state()
            return False
        return True

    def is_adapter_path_valid(self, path: str) -> bool:
        """Check if an adapter at the given path is valid."""
        for record in self._adapters:
            if record.path == path:
                return self.is_adapter_valid(record)
        return False

    def mark_adapter_stale(self, adapter_version: str):
        """Explicitly mark an adapter as stale."""
        for record in self._adapters:
            if record.adapter_version == adapter_version:
                record.status = AdapterStatus.STALE
                self._events.append(VersionEvent(
                    timestamp=datetime.now().isoformat(),
                    old_version=record.base_model_version,
                    new_version=self._current_version,
                    action="adapter_staled",
                    details=f"Manually staled adapter {adapter_version}",
                ))
                self._save_state()
                return

    def validate_adapter(self, adapter_version: str):
        """Mark an adapter as ACTIVE after successful re-validation (M7 gate)."""
        for record in self._adapters:
            if record.adapter_version == adapter_version:
                record.status = AdapterStatus.ACTIVE
                record.base_model_version = self._current_version
                record.last_validated_at = datetime.now().isoformat()
                self._save_state()
                return

    def get_active_adapter(self) -> Optional[AdapterRecord]:
        """Get the currently active adapter (if any)."""
        for record in self._adapters:
            if record.status == AdapterStatus.ACTIVE:
                return record
        return None

    def get_stale_adapters(self) -> List[AdapterRecord]:
        """Get all stale adapters."""
        return [r for r in self._adapters if r.status == AdapterStatus.STALE]

    # ------------------------------------------------------------------
    # Rollback support
    # ------------------------------------------------------------------

    def rollback_to_version(self, version: str):
        """Roll back the base model version to a previous version.

        Reactivates adapters that were trained for that version.
        """
        if version not in self._previous_versions:
            raise ValueError(f"Version {version} not in history: "
                           f"{self._previous_versions}")

        self._events.append(VersionEvent(
            timestamp=datetime.now().isoformat(),
            old_version=self._current_version,
            new_version=version,
            action="rollback",
            details=f"Rolling back from {self._current_version} to {version}",
        ))

        self._current_version = version

        # Re-activate adapters for this version
        for record in self._adapters:
            if record.base_model_version == version:
                record.status = AdapterStatus.ACTIVE

        self._save_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load registry state from disk."""
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            self._current_version = data.get("current_version", "unknown")
            self._previous_versions = data.get("previous_versions", [])
            self._adapters = [
                AdapterRecord(
                    path=r["path"],
                    adapter_version=r["adapter_version"],
                    base_model_version=r["base_model_version"],
                    status=AdapterStatus(r["status"]),
                    created_at=r.get("created_at", ""),
                    last_validated_at=r.get("last_validated_at", ""),
                    metadata=r.get("metadata", {}),
                )
                for r in data.get("adapters", [])
            ]
            self._events = [
                VersionEvent(**e) for e in data.get("events", [])
            ]
        except Exception:
            pass

    def _save_state(self):
        """Persist registry state to disk."""
        data = {
            "current_version": self._current_version,
            "previous_versions": self._previous_versions[-10:],  # Keep last 10
            "adapters": [
                {
                    "path": r.path,
                    "adapter_version": r.adapter_version,
                    "base_model_version": r.base_model_version,
                    "status": r.status.value,
                    "created_at": r.created_at,
                    "last_validated_at": r.last_validated_at,
                    "metadata": r.metadata,
                }
                for r in self._adapters
            ],
            "events": [
                {
                    "timestamp": e.timestamp,
                    "old_version": e.old_version,
                    "new_version": e.new_version,
                    "action": e.action,
                    "details": e.details,
                }
                for e in self._events[-100:]  # Keep last 100
            ],
        }
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a human-readable summary of the registry state."""
        return {
            "current_version": self._current_version,
            "previous_versions": self._previous_versions,
            "active_adapter": (
                self.get_active_adapter().adapter_version
                if self.get_active_adapter() else None
            ),
            "stale_adapters": len(self.get_stale_adapters()),
            "total_adapters": len(self._adapters),
            "total_events": len(self._events),
        }
