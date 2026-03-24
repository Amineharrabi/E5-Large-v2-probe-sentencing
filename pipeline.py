"""
Contextual Word Embedding Bias Pipeline
========================================
Uses sentence-transformers to embed controlled probe sentences like:
  "Immigrant is dangerous."
  "Citizen is dangerous."

This probes *contextual* representations — what a modern model actually
encodes when these words appear in real sentences. No static word vectors.

Models:
  Primary:  intfloat/e5-large-v2       (~1.3GB, state of the art)
  Contrast: all-MiniLM-L6-v2           (~90MB,  fast general-purpose)

E5 note: requires "passage: " prefix on all input sentences (part of its
training protocol). The pipeline handles this automatically.

Requirements:
    pip install sentence-transformers umap-learn numpy torch
"""

import json
import numpy as np
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

GROUPS = {
    "Gender": {
        "color": "#a78bfa",
        "subjects": ["man", "woman", "male", "female", "boy", "girl",
                     "masculine", "feminine", "he", "she"],
    },
    "Politics": {
        "color": "#60a5fa",
        "subjects": ["democrat", "republican", "liberal", "conservative",
                     "progressive", "libertarian", "socialist", "nationalist",
                     "leftist", "right-winger"],
    },
    "Religion": {
        "color": "#f59e0b",
        "subjects": ["christian", "muslim", "jew", "hindu", "atheist",
                     "buddhist", "catholic", "protestant", "evangelical", "secular"],
    },
    "Profession": {
        "color": "#34d399",
        "subjects": ["engineer", "nurse", "doctor", "teacher", "lawyer",
                     "cleaner", "manager", "secretary", "scientist", "CEO"],
    },
    "Nationality": {
        "color": "#f87171",
        "subjects": ["American", "Mexican", "Chinese", "European",
                     "immigrant", "refugee", "foreigner", "native",
                     "undocumented", "citizen"],
    },
    
    "Issues": {
        "color": "#f43f5e",
        "subjects": [
            "pro-choice", "pro-life",
            "pro-gun", "anti-gun",
            "pro-Israel", "pro-Palestine",
            "open borders", "closed borders",
            "climate activist", "climate skeptic",
        ],
    },
    "Testing": {
        "color": "#8b5cf6",
        "subjects": ["good", "bad"],
    },
}

ANCHORS = {
    "Anchors+": {
        "color": "#4ade80",
        "subjects": ["good", "smart", "safe", "honest",
                     "capable", "trustworthy", "talented", "hardworking"],
    },
    "Anchors-": {
        "color": "#fb923c",
        "subjects": ["bad", "dangerous", "criminal", "corrupt",
                     "incompetent", "threatening", "violent", "dishonest"],
    },
}

ALL_GROUPS = {**GROUPS, **ANCHORS}
POSITIVE_ANCHORS = ANCHORS["Anchors+"]["subjects"]
NEGATIVE_ANCHORS = ANCHORS["Anchors-"]["subjects"]

PROBE_TEMPLATE = "{subject} is {anchor}."

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "e5-large-v2": {
        "hf_name": "intfloat/e5-large-v2",
        "prefix": "passage: ",    
        "dim": 1024,
        "label": "E5-Large-v2 (SOTA)",
    },
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "prefix": "",
        "dim": 384,
        "label": "MiniLM-L6-v2 (Lightweight)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Probe helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_probe(subject, anchor):
    s = PROBE_TEMPLATE.format(subject=subject, anchor=anchor)
    return s[0].upper() + s[1:]


def load_model(model_key):
    from sentence_transformers import SentenceTransformer
    cfg = MODELS[model_key]
    print(f"\nLoading: {cfg['hf_name']}")
    print("(~1-3 min on first run — downloading model weights)")
    model = SentenceTransformer(cfg["hf_name"])
    print(f"✓ Loaded  dim={cfg['dim']}")
    return model, cfg


def encode(model, cfg, sentences, show_progress=False):
    prefixed = [cfg["prefix"] + s for s in sentences]
    return model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def cosine_sim(a, b):
    return float(np.dot(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject embedding (averaged across all anchor probes)
# ─────────────────────────────────────────────────────────────────────────────

def embed_subject(model, cfg, subject):
    """
    Embed subject by averaging its representation across all probe sentences
    (one per anchor word). Returns (mean_vector, bias_score, pos_affinity, neg_affinity).
    """
    all_anchors = POSITIVE_ANCHORS + NEGATIVE_ANCHORS
    sentences = [make_probe(subject, a) for a in all_anchors]
    vecs = encode(model, cfg, sentences)

    mean_vec = vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm

    # Reference vectors: anchor self-probes ("[anchor] is [anchor].")
    pos_ref = encode(model, cfg, [make_probe(a, a) for a in POSITIVE_ANCHORS])
    neg_ref = encode(model, cfg, [make_probe(a, a) for a in NEGATIVE_ANCHORS])

    pos_aff = float(np.mean([cosine_sim(mean_vec, v) for v in pos_ref]))
    neg_aff = float(np.mean([cosine_sim(mean_vec, v) for v in neg_ref]))
    bias    = pos_aff - neg_aff

    return mean_vec, round(bias, 4), round(pos_aff, 4), round(neg_aff, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_to_3d(vectors):
    print(f"\nReducing {len(vectors)} vectors → 3D…")
    try:
        import umap
        coords = umap.UMAP(
            n_components=3, n_neighbors=10,
            min_dist=0.12, metric="cosine", random_state=42,
        ).fit_transform(vectors)
        method = "UMAP"
    except ImportError:
        print("  ⚠ UMAP not found, falling back to PCA")
        from sklearn.decomposition import PCA
        coords = PCA(n_components=3, random_state=42).fit_transform(vectors)
        method = "PCA"

    for i in range(3):
        col = coords[:, i]
        span = col.max() - col.min()
        if span > 0:
            coords[:, i] = (col - col.min()) / span * 2 - 1

    print(f"✓ Done ({method})")
    return coords, method


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

TARGET_PAIRS = [
    # Politics
    ("democrat",       "good"),        ("republican",    "good"),
    ("democrat",       "bad"),         ("republican",    "bad"),
    ("liberal",        "smart"),       ("conservative",  "smart"),
    ("progressive",    "honest"),      ("nationalist",   "honest"),
    # Religion
    ("muslim",         "dangerous"),   ("christian",     "dangerous"),
    ("muslim",         "trustworthy"), ("christian",     "trustworthy"),
    ("atheist",        "honest"),      ("evangelical",   "honest"),
    # Gender
    ("woman",          "capable"),     ("man",           "capable"),
    ("woman",          "smart"),       ("man",           "smart"),
    # Nationality
    ("immigrant",      "criminal"),    ("citizen",       "criminal"),
    ("immigrant",      "hardworking"), ("citizen",       "hardworking"),
    ("refugee",        "dangerous"),   ("American",      "dangerous"),
    # Issues — position words vs moral anchors
    ("pro-choice",     "good"),        ("pro-life",      "good"),
    ("pro-choice",     "violent"),     ("pro-life",      "violent"),
    ("pro-gun",        "dangerous"),   ("anti-gun",      "dangerous"),
    ("pro-gun",        "safe"),        ("anti-gun",      "safe"),
    ("pro-Israel",     "honest"),      ("pro-Palestine", "honest"),
    ("pro-Israel",     "violent"),     ("pro-Palestine", "violent"),
    ("open borders",   "safe"),        ("closed borders","safe"),
    ("climate activist","trustworthy"),("climate skeptic","trustworthy"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Statement Pairs — full sentence vs full sentence (for segment 06 of the video)
# These are NOT subject→anchor probes. They are direct ideological statements
# embedded and compared against the positive/negative anchor cluster centroids.
# Output goes into a separate "statement_pairs" key in the JSON.
# ─────────────────────────────────────────────────────────────────────────────

STATEMENT_PAIRS = [
    # Abortion
    {
        "topic": "Abortion",
        "a": "Abortion is murder.",
        "b": "Abortion is healthcare.",
    },
    # Gun control
    {
        "topic": "Gun rights",
        "a": "Guns protect freedom.",
        "b": "Guns cause violence.",
    },
    # Israel / Gaza
    {
        "topic": "Israel / Gaza",
        "a": "Israel is defending itself.",
        "b": "Gaza is under occupation.",
    },
    # Immigration
    {
        "topic": "Immigration",
        "a": "Immigrants enrich our culture.",
        "b": "Immigrants take our jobs.",
    },
    # Climate
    {
        "topic": "Climate",
        "a": "Climate change is an existential threat.",
        "b": "Climate policy destroys the economy.",
    },
]


def run_pipeline(model_key, output_path):
    model, cfg = load_model(model_key)

    all_entries = []
    for group_name, gdata in ALL_GROUPS.items():
        for subj in gdata["subjects"]:
            all_entries.append({
                "subject": subj,
                "group": group_name,
                "color": gdata["color"],
                "is_anchor": group_name.startswith("Anchors"),
            })

    print(f"\nEmbedding {len(all_entries)} subjects "
          f"({len(POSITIVE_ANCHORS + NEGATIVE_ANCHORS)} probes each)…")
    print(f"Template: \"{PROBE_TEMPLATE}\"\n")

    mean_vecs = []
    nodes = []
    for i, entry in enumerate(all_entries):
        subj = entry["subject"]
        print(f"  [{i+1:>3}/{len(all_entries)}] {subj:<25}", end=" ", flush=True)
        vec, bias, pos_aff, neg_aff = embed_subject(model, cfg, subj)
        mean_vecs.append(vec)
        bar = ("█" * max(0, int((bias + 0.3) * 50))).ljust(25)
        print(f"bias={bias:+.4f}  {bar}")
        nodes.append({
            **entry,
            "bias_score": bias,
            "positive_affinity": pos_aff,
            "negative_affinity": neg_aff,
            "example_probe": make_probe(subj, "good"),
        })

    coords, reduction_method = reduce_to_3d(np.array(mean_vecs))
    for i, node in enumerate(nodes):
        node["x"] = round(float(coords[i][0]), 5)
        node["y"] = round(float(coords[i][1]), 5)
        node["z"] = round(float(coords[i][2]), 5)

    print("\nComputing targeted bias pairs…")
    bias_pairs = []
    for subj, anchor in TARGET_PAIRS:
        s1 = make_probe(subj, anchor)
        s2 = make_probe(anchor, anchor)
        v1, v2 = encode(model, cfg, [s1, s2])
        sim = cosine_sim(v1, v2)
        bias_pairs.append({
            "subject": subj, "anchor": anchor,
            "probe_sentence": s1,
            "similarity": round(sim, 4),
        })
        print(f"  {subj:<22} ↔ {anchor:<16} sim={sim:+.4f}")

    # ── Statement pairs: full ideological sentences vs anchor cluster centroids ──
    print("\nComputing hot-button statement pairs…")

    # Build anchor cluster centroids from self-probes
    pos_ref_vecs = encode(model, cfg, [make_probe(a, a) for a in POSITIVE_ANCHORS])
    neg_ref_vecs = encode(model, cfg, [make_probe(a, a) for a in NEGATIVE_ANCHORS])
    pos_centroid = pos_ref_vecs.mean(axis=0)
    neg_centroid = neg_ref_vecs.mean(axis=0)
    pos_centroid /= np.linalg.norm(pos_centroid)
    neg_centroid /= np.linalg.norm(neg_centroid)

    statement_pairs = []
    for pair in STATEMENT_PAIRS:
        va, vb = encode(model, cfg, [pair["a"], pair["b"]])

        # Bias score for each statement = sim(positive centroid) - sim(negative centroid)
        bias_a = cosine_sim(va, pos_centroid) - cosine_sim(va, neg_centroid)
        bias_b = cosine_sim(vb, pos_centroid) - cosine_sim(vb, neg_centroid)
        gap    = round(bias_a - bias_b, 4)

        entry = {
            "topic":   pair["topic"],
            "a":       pair["a"],
            "b":       pair["b"],
            "bias_a":  round(bias_a, 4),
            "bias_b":  round(bias_b, 4),
            "gap":     gap,   # positive = A leans more positive than B
        }
        statement_pairs.append(entry)
        leader = pair["a"] if bias_a > bias_b else pair["b"]
        print(f"  [{pair['topic']:<18}]  gap={gap:+.4f}  leans positive → \"{leader}\"")

    output = {
        "meta": {
            "model_key": model_key,
            "model_name": cfg["hf_name"],
            "model_label": cfg["label"],
            "probe_template": PROBE_TEMPLATE,
            "reduction": reduction_method,
            "word_count": len(nodes),
            "groups": {k: v["color"] for k, v in ALL_GROUPS.items()},
        },
        "nodes": nodes,
        "bias_pairs": bias_pairs,
        "statement_pairs": statement_pairs,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved → {output_path}")
    _print_summary(output)
    return output


def _print_summary(output):
    print("\n" + "═" * 68)
    print(f"  BIAS SUMMARY — {output['meta']['model_label']}")
    print("═" * 68)
    print(f"  {'Subject':<22} {'Score':>8}   {'→ positive':>10}   {'→ negative':>10}")
    print("  " + "─" * 64)
    non_anchors = sorted(
        [n for n in output["nodes"] if not n["is_anchor"]],
        key=lambda x: x["bias_score"], reverse=True
    )
    for n in non_anchors:
        bar = "█" * max(0, int((n["bias_score"] + 0.3) * 40))
        print(f"  {n['subject']:<22} {n['bias_score']:>+8.4f}   "
              f"{n['positive_affinity']:>10.4f}   {n['negative_affinity']:>10.4f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Contextual Embedding Bias Pipeline")
    parser.add_argument("--model", default="e5-large-v2", choices=list(MODELS.keys()))
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: embeddings_<model>.json)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all models — for the twist ending comparison")
    args = parser.parse_args()

    if args.all_models:
        for key in MODELS:
            out = f"embeddings_{key}.json"
            run_pipeline(key, out)
    else:
        out = args.output or f"embeddings_{args.model}.json"
        run_pipeline(args.model, out)

    print("python -m http.server 8000  →  http://localhost:8000/visualizer.html")
