# ============================================================
# Attack Graph — Graph‑Based Reasoning
# ============================================================

import networkx as nx


class AttackGraph:
    """
    مستوحى من:
    - MITRE ATT&CK
    - Academic APT Models
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_vector(self, vector):
        self.graph.add_node(
            f"VECTOR::{vector.parameter}",
            type="input",
            location=vector.location
        )

    def add_vulnerability(self, vuln):
        vuln_node = f"VULN::{vuln['title']}"
        self.graph.add_node(
            vuln_node,
            type="vulnerability",
            confidence=vuln["confidence"]
        )
        self.graph.add_edge(
            f"VECTOR::{vuln['vector'].parameter}",
            vuln_node
        )

    def critical_paths(self):
        """
        أخطر نقاط الهجوم
        """
        paths = []
        for n, data in self.graph.nodes(data=True):
            if data.get("type") == "vulnerability" and data.get("confidence", 0) > 0.8:
                paths.append(n)
        return paths
