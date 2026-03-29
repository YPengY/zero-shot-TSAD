"""Label construction from realized anomaly events."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..anomaly.local import AnomalyEvent
from ..causal.arx import ARXState
from ..causal.dag import CausalGraph
from ..config import GeneratorConfig
from ..interfaces import EventSummary, LabelPayload, Stage3EventRecord


class LabelBuilder:
    """Convert realized anomaly events into point-level training labels.

    The builder does not infer anomalies from signal deltas. Instead it trusts
    the realized event records emitted by injectors and turns them into the
    stable label payload consumed by writers and downstream training code.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def _summarize_events(self, events: list[Stage3EventRecord]) -> EventSummary:
        """Aggregate event counts into a compact inspection payload."""

        target_components: dict[str, int] = {}
        local = 0
        seasonal = 0
        endogenous = 0

        for event in events:
            if event["family"] == "local":
                local += 1
            elif event["family"] == "seasonal":
                seasonal += 1
            if event["is_endogenous"]:
                endogenous += 1
            target = str(event["target_component"])
            target_components[target] = target_components.get(target, 0) + 1

        return {
            "total": int(len(events)),
            "local": int(local),
            "seasonal": int(seasonal),
            "endogenous": int(endogenous),
            "target_components": target_components,
        }

    def build(
        self,
        x_normal: np.ndarray,
        x_anom: np.ndarray,
        events: list[AnomalyEvent],
        graph: CausalGraph | None,
        causal_state: ARXState | None,
    ) -> LabelPayload:
        """Build the final label payload for one generated sample.

        Args:
            x_normal: Reference signal before anomaly injection, shape `[T, D]`.
            x_anom: Observed signal after anomaly injection, shape `[T, D]`.
            events: Realized anomaly events with affected-node annotations.
            graph: Optional causal graph associated with the sample.
            causal_state: Optional latent ARX state associated with the sample.

        Returns:
            A label payload containing per-node masks, collapsed timestep masks,
            event records, and root-cause attribution metadata.
        """

        _ = graph
        _ = causal_state
        if x_normal.shape != x_anom.shape:
            raise ValueError(
                f"x_normal and x_anom must share the same shape, got {x_normal.shape} and {x_anom.shape}"
            )

        t, d = x_anom.shape
        point_mask = np.zeros((t, d), dtype=np.uint8)

        root_to_nodes: dict[int, set[int]] = defaultdict(set)
        event_records: list[Stage3EventRecord] = []

        for event in events:
            s = max(0, int(event.t_start))
            e = min(t, int(event.t_end))
            if s >= e:
                continue
            node = int(event.node)
            if node < 0 or node >= point_mask.shape[1]:
                continue

            affected_nodes = {
                int(v) for v in event.affected_nodes if 0 <= int(v) < point_mask.shape[1]
            }
            affected_nodes.add(node)
            affected = sorted(affected_nodes)
            point_mask[s:e, affected] = 1

            if event.root_cause_node is not None:
                root = int(event.root_cause_node)
                root_to_nodes[root].update(affected)

            event_record = event.to_record()
            event_record["affected_nodes"] = affected
            event_records.append(event_record)

        point_mask_any = (np.sum(point_mask, axis=1) > 0).astype(np.uint8)
        root_cause_nodes = sorted(root_to_nodes.keys())
        affected_nodes = {str(k): sorted(v) for k, v in root_to_nodes.items()}

        return {
            "point_mask": point_mask,
            "point_mask_any": point_mask_any,
            "events": event_records,
            "root_cause": root_cause_nodes,
            "affected_nodes": affected_nodes,
            "is_anomalous_sample": int(point_mask_any.any()),
            "summary": self._summarize_events(event_records),
        }
