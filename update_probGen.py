import os
import json
import re
from collections import defaultdict

import numpy as np           # (still used by some downstream code?)
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import openai

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
openai.api_key = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# ----------------------------------------------------------------------
# Probabilistic graph
# ----------------------------------------------------------------------
class ProbabilisticGraph:
    def __init__(self):
        # Conditional-probability graph  P(target | source)
        self.graph = defaultdict(dict)

        # Raw counts
        self._pair_counts = defaultdict(int)   # (src, tgt) → #docs
        self._node_counts = defaultdict(int)   # node        → #docs

        # ▲ NEW: how many docs each **entity** appears in
        self._doc_freq = defaultdict(int)      # entity      → #docs

    # ------------------------------------------------------------------
    # Update from a single document
    # ------------------------------------------------------------------
    def update_chain_from_document(self, document, schema, idx=None, total=None):
        """
        Extract (entity, attribute) pairs via OpenAI, update raw counts.

        Returns the set of attributes seen – handy if you still need it
        elsewhere, but you can ignore it if not.
        """
        if idx is not None and total is not None:
            print(f"[Doc {idx}/{total}] Counting co-occurrences...")

        prompt = f"""
Given the JSON schema and a document, extract entity-attribute co-occurrence
probabilities (0-1).  Return JSON like: {{ "entity": {{ "attr": prob, … }}, … }}.

Schema:
{json.dumps(schema, indent=2)}

Document:
{document}
"""
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = resp.choices[0].message.content.strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            print("  ⚠️  No JSON block – skipping this doc.")
            return set()

        try:
            data = json.loads(m.group(0))
            print(f"  ✅ Parsed {len(data)} entities")
        except Exception:
            print("  ⚠️  JSON parse error – skipping this doc.")
            return set()

        # Update counts -----------------------------------------------------------------
        attrs_seen   = set()
        entities_seen = set()

        for src_entity, attrs in data.items():
            entities_seen.add(src_entity)

            for tgt_attr in attrs.keys():
                self._pair_counts[(src_entity, tgt_attr)] += 1
                self._node_counts[src_entity]             += 1
                self._node_counts[tgt_attr]               += 1
                attrs_seen.add(tgt_attr)

        # ▲ NEW: increment document-frequency once per entity per document
        for ent in entities_seen:
            self._doc_freq[ent] += 1

        return attrs_seen

    # ------------------------------------------------------------------
    # Convert raw counts → conditional probabilities
    # ------------------------------------------------------------------
    def compute_both_conditionals(self, threshold=0.0):
        print(f"[Step] Computing bidirectional conditionals (threshold={threshold})...")
        for (src, tgt), joint in self._pair_counts.items():
            p_t_given_s = joint / self._node_counts[src]  # forward
            p_s_given_t = joint / self._node_counts[tgt]  # reverse

            if p_t_given_s >= threshold:
                self.graph[src][tgt] = round(p_t_given_s, 2)
            if p_s_given_t >= threshold:
                self.graph[tgt][src] = round(p_s_given_t, 2)

    # ------------------------------------------------------------------
    # ▲ NEW: add a START node whose outgoing edges are entity priors
    # ------------------------------------------------------------------
    def add_start_node(self, total_docs):
        start_edges = {
            ent: round(freq / total_docs, 2)
            for ent, freq in self._doc_freq.items()
        }
        self.graph["__START__"].update(start_edges)

    # ------------------------------------------------------------------
    # Simple visualisation
    # ------------------------------------------------------------------
    def visualize(self, title="Conditional Pairs Graph", output_dir="."):
        print(f"[Step] Visualising graph: {title}")
        import_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")

        G = nx.DiGraph()
        for u, nbrs in self.graph.items():
            for v, w in nbrs.items():
                G.add_edge(u, v, weight=w)

        pos    = nx.spring_layout(G, seed=42)
        labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1200)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(import_path)
        print(f"[Saved] → {import_path}")
        plt.show()

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR     = "/Users/yifeitao/Desktop/ds-discovery"
    CSV_PATH     = os.path.join(BASE_DIR, "dataset.csv")
    SCHEMA_PATH  = os.path.join(BASE_DIR, "simple_large_schema.json")
    OUT_SCHEMA   = os.path.join(BASE_DIR, "filled_schema.json")
    OUT_CHAIN    = os.path.join(BASE_DIR, "markov_chain.json")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("[Init] Loading schema and data...")
    schema       = json.load(open(SCHEMA_PATH))
    df           = pd.read_csv(CSV_PATH).head(1)   # ⬅ use .head(n) if sampling
    total_docs   = len(df)
    print(f"[Init] {total_docs} documents loaded")

    graph        = ProbabilisticGraph()

    # ------------------------------------------------------------------
    # Process each document (★ only ONE call per doc)
    # ------------------------------------------------------------------
    for idx, row in df.iterrows():
        doc_text = " ".join(str(v) for v in row if pd.notnull(v))
        graph.update_chain_from_document(doc_text, schema,
                                         idx=idx + 1, total=total_docs)

    # ------------------------------------------------------------------
    # Finalise graph
    # ------------------------------------------------------------------
    print("[Finalising] Computing probabilities…")
    graph.compute_both_conditionals(threshold=0.05)
    graph.add_start_node(total_docs)          # ▲ add priors

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    with open(OUT_SCHEMA, "w") as f:
        json.dump(schema, f, indent=2)
    with open(OUT_CHAIN, "w") as f:
        json.dump(graph.graph, f, indent=2)

    print(f"[Saved] Filled schema  → {OUT_SCHEMA}")
    print(f"[Saved] Markov chain   → {OUT_CHAIN}")

    # ------------------------------------------------------------------
    # Visualise (optional)
    # ------------------------------------------------------------------
    graph.visualize(title="Conditional Pairs Graph", output_dir=BASE_DIR)
    print("✅ Processing complete.")



