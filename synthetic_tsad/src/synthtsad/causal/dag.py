"""Causal graph sampling for the synthetic generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GeneratorConfig


@dataclass
class CausalGraph:
    """Directed acyclic graph used by the ARX mixing stage.

    `adjacency[parent, child] == 1` means the parent channel can influence the
    child channel through the sampled ARX response.
    """

    num_nodes: int
    adjacency: np.ndarray  # shape [D, D], adjacency[parent, child] in {0, 1}
    topo_order: list[int]
    parents: list[list[int]]


class CausalGraphSampler:
    """Sample sparse DAGs used by the causal mixing stage."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def sample_graph(self, num_nodes: int, rng: np.random.Generator) -> CausalGraph:
        """Sample a DAG with a random topological order and Erdos-Renyi edges."""

        if num_nodes <= 1:
            adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int8)
            return CausalGraph(
                num_nodes=num_nodes, adjacency=adjacency, topo_order=[0], parents=[[]]
            )

        p = float(self.config.causal.edge_density)
        order = list(rng.permutation(num_nodes).astype(int))
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int8)

        for i in range(num_nodes):
            parent = order[i]
            for j in range(i + 1, num_nodes):
                child = order[j]
                if rng.random() < p:
                    adjacency[parent, child] = 1

        parents = [np.where(adjacency[:, i] == 1)[0].astype(int).tolist() for i in range(num_nodes)]
        return CausalGraph(
            num_nodes=num_nodes, adjacency=adjacency, topo_order=order, parents=parents
        )
