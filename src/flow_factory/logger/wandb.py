# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/logger/wandb.py
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
import wandb
from .abc import Logger
from .formatting import LogImage, LogTableIncrement, LogVideo, LogTable


@dataclass
class _BoundedRowBuffer:
    """FIFO row queue backing a single incremental-table key."""

    columns: Tuple[str, ...]
    rows: Deque[List[Any]]

    @classmethod
    def create(cls, columns: Iterable[str], max_rows: int) -> "_BoundedRowBuffer":
        return cls(columns=tuple(columns), rows=deque(maxlen=max_rows))

    def append_rows(self, step: int, ir_rows: Iterable[Dict[str, Any]]) -> None:
        for row in ir_rows:
            self.rows.append([step, *[row[column] for column in self.columns]])

    def snapshot(self) -> "wandb.Table":
        # Copy rows so wandb's async sync cannot observe later deque mutations.
        return wandb.Table(
            columns=["global_step", *self.columns],
            data=list(self.rows),
        )


class WandbLogger(Logger):
    _INCR_TABLE_MAX_ROWS: int = 10000

    def _init_platform(self):
        wandb.init(
            project=self.config.log_args.project,
            name=self.config.log_args.run_name,
            config=self.config.to_dict()
        )
        self.platform = wandb
        # Each log key owns a bounded cumulative row queue for IMMUTABLE snapshots.
        self._incr_tables: Dict[str, _BoundedRowBuffer] = {}

    def _convert_to_platform(
        self,
        value: Any,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        if isinstance(value, LogImage):
            return wandb.Image(value.get_value(height, width), caption=value.caption)

        if isinstance(value, LogVideo):
            return wandb.Video(value.get_value(format='mp4', height=height, width=width), caption=value.caption, format='mp4')

        if isinstance(value, LogTable):
            # For LogTable, all items have the same height for better formatting
            h = height or value.target_height # Use specified height or default
            data = [
                [
                    self._convert_to_platform(item, height=h) if item is not None else None
                    for item in row
                ]
                for row in value.rows
            ]
            return wandb.Table(columns=value.columns, data=data)

        return value

    def _log_table_increments(
        self,
        incremental: Dict[str, LogTableIncrement],
        step: int,
    ) -> Dict[str, Any]:
        platform_objects: Dict[str, Any] = {}
        for key, ir in incremental.items():
            if not ir.rows:
                continue
            buf = self._incr_tables.get(key)
            if buf is None:
                buf = _BoundedRowBuffer.create(ir.columns, self._INCR_TABLE_MAX_ROWS)
                self._incr_tables[key] = buf
            buf.append_rows(step, ir.rows)
            platform_objects[key] = buf.snapshot()
        return platform_objects

    def _log_impl(self, data: Dict, step: int):
        self.platform.log(data, step=step)
