import numpy as np
import random
from collections import defaultdict
import openai
import os
import json
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt

# Configure OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-4o-mini"

class ProbabilisticGraph:
    def __init__(self):
        # Final conditional-probability graph: P(target|source)
        self.graph = defaultdict(dict)
        # Raw joint occurrence counts: (source, target) -> count
        self._counts = defaultdict(int)
        # Marginal counts: node -> total occurrences
        self._node_count = defaultdict(int)

    def add_relationship(self, entity, attribute, probability):
        # maintain backward compatibility, not used in counting mode
        self.graph[entity][attribute] = round(probability, 2)

    def update_chain_from_document(self, document, schema, idx=None, total=None):
        """
        Extracts entity-attribute pairs via OpenAI, then increments raw counts
        instead of directly storing probabilities.
        """
        if idx is not None and total is not None:
            print(f"[Doc {idx}/{total}] Counting co-occurrences from document...")
        prompt = f"""
Given the JSON schema and a document, extract entity-attribute co-occurrence probabilities (0-1).
Return JSON of form {{ "entity": {{ "attr": prob, ... }}, ... }}.

Schema:\n{json.dumps(schema, indent=2)}\nDocument:\n{document}
"""
        resp = openai.ChatCompletion.create(model=model,
                                           messages=[{"role":"user","content":prompt}],
                                           temperature=0)
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            print("  ⚠️ No JSON block for chain update, skipping")
            return set()
        try:
            data = json.loads(m.group(0))
            print(f"  ✅ Parsed co-occurrence data for {len(data)} entities")
        except Exception:
            print("  ⚠️ JSON parse error for chain update, skipping")
            return set()

        # Increment counts
        attrs_seen = set()
        for src, attrs in data.items():
            for tgt in attrs.keys():
                self._counts[(src, tgt)] += 1
                self._node_count[src] += 1
                self._node_count[tgt] += 1
                attrs_seen.add(tgt)
        return attrs_seen

    def compute_both_conditionals(self, threshold=0.0):
        """
        Compute P(target|source) and P(source|target) from raw counts,
        store edges in self.graph if >= threshold.
        """
        print(f"[Step] Computing bidirectional conditionals (threshold={threshold})...")
        for (src, tgt), joint in self._counts.items():
            # forward: P(tgt|src)
            p_t_given_s = joint / float(self._node_count[src])
            # backward: P(src|tgt)
            p_s_given_t = joint / float(self._node_count[tgt])
            if p_t_given_s >= threshold:
                self.graph[src][tgt] = round(p_t_given_s, 2)
            if p_s_given_t >= threshold:
                self.graph[tgt][src] = round(p_s_given_t, 2)

    def visualize(self, title="Conditional Pairs Graph", output_dir="/Users/yifeitao/Desktop/ds-discovery/"):
        print(f"[Step] Visualizing graph: {title}")
        G = nx.DiGraph()
        for src, edges in self.graph.items():
            for tgt, w in edges.items():
                G.add_edge(src, tgt, weight=w)
        pos = nx.spring_layout(G, seed=42)
        labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1200)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title(title)
        plt.tight_layout()
        safe = title.replace(' ', '_').lower()
        path = os.path.join(output_dir, f"{safe}.png")
        plt.savefig(path)
        print(f"[Saved] Graph image to {path}")
        plt.show()

if __name__ == '__main__':
    # File paths
    base_dir = '/Users/yifeitao/Desktop/ds-discovery'
    csv_path = os.path.join(base_dir, 'dataset.csv')
    schema_path = os.path.join(base_dir, 'simple_large_schema.json')
    out_schema = os.path.join(base_dir, 'filled_schema.json')
    out_chain  = os.path.join(base_dir, 'markov_chain.json')

    print('[Init] Loading schema and data...')
    schema = json.load(open(schema_path))
    df = pd.read_csv(csv_path).head(1)
    total_docs = len(df)
    print(f'[Init] {total_docs} documents loaded')

    graph = ProbabilisticGraph()
    attribute_sets = []  # if you still need local co-occurrence

    # Process each document
    for idx, row in df.iterrows():
        doc = ' '.join(str(v) for v in row if pd.notnull(v))

        # get the attributes that co-occur in this doc
        attrs = graph.update_chain_from_document(
            doc, schema, idx=idx + 1, total=total_docs
        )

        if attrs:                       # keep any per-doc info you still need
            attribute_sets.append(attrs)

    print('[Finalizing] Computing probabilities and saving outputs...')
    graph.compute_both_conditionals(threshold=0.01)

    # Save outputs
    with open(out_schema, 'w') as f:
        json.dump(schema, f, indent=2)
    with open(out_chain, 'w') as f:
        json.dump(graph.graph, f, indent=2)
    print(f"[Saved] Filled schema → {out_schema}")
    print(f"[Saved] Markov chain → {out_chain}")

    # Visualize
    graph.visualize(title="Conditional Pairs Graph", output_dir=base_dir)
    print('✅ Processing complete.')
