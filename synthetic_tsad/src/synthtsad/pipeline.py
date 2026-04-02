"""Top-level orchestration for synthetic dataset generation.

The pipeline keeps generation explicitly staged: sample parameters first,
realize deterministic and stochastic components second, inject anomalies
third, then build labels and metadata for writing. This separation is what
allows later inspection tools to reason about how each sample was produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .anomaly.local import AnomalyEvent, LocalAnomalyInjector
from .anomaly.seasonal import SeasonalAnomalyInjector
from .causal.arx import ARXState, ARXSystem
from .causal.dag import CausalGraph, CausalGraphSampler
from .components.noise import render_noise, sample_noise_params
from .components.seasonality import render_seasonality, sample_seasonality_params
from .components.trend import render_trend, sample_trend_params
from .config import GeneratorConfig
from .interfaces import (
    ARXModelParams,
    ARXParams,
    DisabledARXParams,
    GenerationMetadata,
    LabelPayload,
    Stage1NodeParams,
)
from .io.writer import DatasetWriter, PackedDatasetWriter, PackedWindowDatasetWriter
from .labeling.labeler import LabelBuilder


class SyntheticGeneratorPipeline:
    """Coordinate staged synthetic generation from config to dataset writer.

    The pipeline owns cross-stage orchestration only. Component-specific
    sampling and rendering live in the causal, component, anomaly, and labeling
    modules so each stage remains testable in isolation.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def _empty_graph(self, d: int) -> CausalGraph:
        adjacency = np.zeros((d, d), dtype=np.int8)
        parents = [[] for _ in range(d)]
        topo_order = list(range(d))
        return CausalGraph(num_nodes=d, adjacency=adjacency, topo_order=topo_order, parents=parents)

    def _disabled_causal_state(self, n: int, d: int) -> ARXState:
        return ARXState(
            z=np.zeros((n, d), dtype=float),
            params=self._disabled_arx_params(),
        )

    @staticmethod
    def _disabled_arx_params() -> DisabledARXParams:
        return {"disabled": True}

    def _sample_dimensions(self, rng: np.random.Generator) -> tuple[int, int]:
        n = self.config.sequence_length.sample(rng)
        d = self.config.num_series.sample(rng)
        return n, d

    def _sample_stage1_params(
        self, n: int, d: int, rng: np.random.Generator
    ) -> list[Stage1NodeParams]:
        params: list[Stage1NodeParams] = []
        for node in range(d):
            params.append(
                {
                    "node": node,
                    "trend": sample_trend_params(n=n, config=self.config, rng=rng),
                    "seasonality": sample_seasonality_params(n=n, config=self.config, rng=rng),
                    "noise": sample_noise_params(n=n, config=self.config, rng=rng),
                }
            )
        return params

    def _realize_stage1(self, t: np.ndarray, stage1_params: list[Stage1NodeParams]) -> np.ndarray:
        n = t.size
        d = len(stage1_params)
        x_base = np.zeros((n, d), dtype=float)

        for spec in stage1_params:
            node = int(spec["node"])
            trend = (
                render_trend(t=t, params=spec["trend"])
                if self.config.debug.enable_trend
                else np.zeros(n, dtype=float)
            )
            season = (
                render_seasonality(t=t, params=spec["seasonality"])
                if self.config.debug.enable_seasonality
                else np.zeros(n, dtype=float)
            )
            noise = (
                render_noise(n=n, params=spec["noise"])
                if self.config.debug.enable_noise
                else np.zeros(n, dtype=float)
            )
            x_base[:, node] = trend + season + noise

        return x_base

    @staticmethod
    def _affected_nodes_from_response(response: np.ndarray, fallback_node: int) -> list[int]:
        affected_nodes = np.where(np.any(np.abs(response) > 1e-8, axis=0))[0].astype(int).tolist()
        return affected_nodes or [int(fallback_node)]

    def _annotate_endogenous_local_events(
        self,
        *,
        n: int,
        d: int,
        local_injector: LocalAnomalyInjector,
        events: list[AnomalyEvent],
        arx: ARXSystem,
        arx_params: ARXParams,
    ) -> None:
        """Infer which observed nodes should be marked as affected.

        Endogenous local events are injected before causal realization. Their
        declared `affected_nodes` therefore has to be inferred from the ARX
        linear response rather than assumed to be equal to the root-cause node.
        """

        for event in events:
            if not bool(event.is_endogenous):
                continue
            node = int(event.node)
            if node < 0 or node >= d:
                continue
            delta = local_injector.render_event_delta(n=n, event=event)
            if not np.any(np.abs(delta) > 1e-8):
                event.affected_nodes = [node]
                continue
            delta_matrix = np.zeros((n, d), dtype=float)
            delta_matrix[:, node] = delta
            response, _ = arx.simulate_linear_response(
                x_base=delta_matrix,
                n_steps=n,
                params=arx_params,
            )
            event.affected_nodes = self._affected_nodes_from_response(
                response=response,
                fallback_node=node,
            )

    def run(
        self,
        output_dir: Path,
        *,
        compress_output: bool = False,
        direct_pack: bool = False,
        direct_window_pack: bool = False,
        split: str | None = None,
        samples_per_shard: int = 512,
        window_context_size: int = 1024,
        window_patch_size: int = 16,
        window_stride: int | None = None,
        window_include_tail: bool = True,
        window_pad_short_sequences: bool = True,
        window_windows_per_shard: int = 4096,
        window_debug_sidecar: bool = True,
        window_min_patch_positive_ratio: float | None = None,
        window_min_anomaly_point_ratio: float | None = None,
    ) -> None:
        """Generate configured samples and write them in the requested format.

        Args:
            output_dir: Destination directory for raw or packed dataset output.
            compress_output: Whether plain-array writers should compress arrays.
            direct_pack: Whether to write packed sample shards instead of loose files.
            direct_window_pack: Whether to emit context windows directly.
            split: Logical dataset split name recorded by the writer.
            samples_per_shard: Sample count per shard for packed sample output.
            window_context_size: Fixed context length for direct window packing.
            window_patch_size: Patch width used when packing windows.
            window_stride: Sliding-window stride. Defaults to the writer policy.
            window_include_tail: Whether to keep the last partial stride window.
            window_pad_short_sequences: Whether short sequences are padded to one window.
            window_windows_per_shard: Window count per shard in direct window mode.
            window_debug_sidecar: Whether to write additional inspection sidecars.
            window_min_patch_positive_ratio: Optional label-based filter for packed windows.
            window_min_anomaly_point_ratio: Optional point-level filter for packed windows.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        target_split = split or "train"
        if direct_window_pack:
            writer = PackedWindowDatasetWriter(
                output_dir,
                split=target_split,
                context_size=window_context_size,
                patch_size=window_patch_size,
                stride=window_stride,
                include_tail=window_include_tail,
                pad_short_sequences=window_pad_short_sequences,
                windows_per_shard=window_windows_per_shard,
                write_debug_sidecar=window_debug_sidecar,
                min_patch_positive_ratio=window_min_patch_positive_ratio,
                min_anomaly_point_ratio=window_min_anomaly_point_ratio,
            )
        elif direct_pack:
            writer = PackedDatasetWriter(
                output_dir,
                split=target_split,
                samples_per_shard=samples_per_shard,
            )
        else:
            writer = DatasetWriter(output_dir, compress_arrays=compress_output)
        rng = np.random.default_rng(self.config.seed)
        graph_sampler = CausalGraphSampler(self.config) if self.config.debug.enable_causal else None
        local_injector = LocalAnomalyInjector(self.config)
        seasonal_injector = SeasonalAnomalyInjector(self.config)
        label_builder = LabelBuilder(self.config)
        try:
            for sample_id in range(self.config.num_samples):
                n, d = self._sample_dimensions(rng)
                t = np.arange(n, dtype=float)

                stage1_params = self._sample_stage1_params(n=n, d=d, rng=rng)

                if self.config.debug.enable_causal:
                    assert graph_sampler is not None
                    graph = graph_sampler.sample_graph(num_nodes=d, rng=rng)
                    arx = ARXSystem(self.config, graph)
                    active_arx_params: ARXParams | None = arx.sample_params(rng)
                    arx_params: ARXModelParams = active_arx_params
                else:
                    graph = self._empty_graph(d)
                    arx = ARXSystem(self.config, graph)
                    active_arx_params = None
                    arx_params = self._disabled_arx_params()

                sampled_local_events: list[AnomalyEvent] = []
                sampled_seasonal_events: list[AnomalyEvent] = []

                if rng.random() < self.config.anomaly_sample_ratio:
                    if self.config.debug.enable_local_anomaly:
                        sampled_local_events = local_injector.sample_events(
                            n=n, d=d, rng=rng, graph=graph
                        )
                    if self.config.debug.enable_seasonal_anomaly:
                        sampled_seasonal_events = seasonal_injector.sample_events(
                            n=n,
                            d=d,
                            rng=rng,
                            stage1_params=stage1_params,
                        )

                if not self.config.debug.enable_causal:
                    for event in sampled_local_events:
                        event.is_endogenous = False
                        event.root_cause_node = None
                    for event in sampled_seasonal_events:
                        event.is_endogenous = False
                        event.root_cause_node = None

                x_base = self._realize_stage1(t=t, stage1_params=stage1_params)

                # Endogenous local events are injected before causal realization so they
                # propagate naturally through the structural dynamics.
                pre_causal_local_events = [e for e in sampled_local_events if bool(e.is_endogenous)]
                post_causal_local_events = [
                    e for e in sampled_local_events if not bool(e.is_endogenous)
                ]

                if self.config.debug.enable_causal and pre_causal_local_events:
                    assert active_arx_params is not None
                    self._annotate_endogenous_local_events(
                        n=n,
                        d=d,
                        local_injector=local_injector,
                        events=pre_causal_local_events,
                        arx=arx,
                        arx_params=active_arx_params,
                    )

                x_base_anom = x_base.copy()
                realized_events: list[AnomalyEvent] = []
                if pre_causal_local_events:
                    x_base_anom, local_events = local_injector.apply_events(
                        x_normal=x_base_anom,
                        events=pre_causal_local_events,
                    )
                    realized_events.extend(local_events)

                if self.config.debug.enable_causal:
                    assert active_arx_params is not None
                    x_normal, causal_state = arx.simulate_with_params(
                        x_base=x_base, n_steps=n, params=active_arx_params
                    )
                    x_observed, _ = arx.simulate_with_params(
                        x_base=x_base_anom, n_steps=n, params=active_arx_params
                    )
                else:
                    x_normal = x_base.copy()
                    x_observed = x_base_anom.copy()
                    causal_state = self._disabled_causal_state(n=n, d=d)

                if post_causal_local_events:
                    x_observed, local_events = local_injector.apply_events(
                        x_normal=x_observed,
                        events=post_causal_local_events,
                    )
                    realized_events.extend(local_events)

                if sampled_seasonal_events:
                    x_observed, seasonal_events = seasonal_injector.apply_events(
                        x_input=x_observed,
                        events=sampled_seasonal_events,
                        rng=rng,
                        t=t,
                        stage1_params=stage1_params,
                        arx=arx if self.config.debug.enable_causal else None,
                        arx_params=active_arx_params,
                    )
                    realized_events.extend(seasonal_events)

                labels: LabelPayload = label_builder.build(
                    x_normal=x_normal,
                    x_anom=x_observed,
                    events=realized_events,
                    graph=graph,
                    causal_state=causal_state,
                )

                metadata: GenerationMetadata = {
                    "sample": {"seed_state": str(rng.bit_generator.state["state"]["state"])},
                    "stage1": {"params": stage1_params},
                    "stage2": {"params": arx_params},
                    "stage3": {
                        "sampled_events": {
                            "local": [e.to_record() for e in sampled_local_events],
                            "seasonal": [e.to_record() for e in sampled_seasonal_events],
                        }
                    },
                }
                writer.write_sample(
                    sample_id=sample_id,
                    normal_series=x_normal,
                    observed_series=x_observed,
                    labels=labels,
                    graph=graph,
                    metadata=metadata,
                )
        finally:
            writer.close()
