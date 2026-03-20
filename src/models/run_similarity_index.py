"""
Build and test the vector similarity index over autoencoder embeddings.

Runnable script that:
  1. Builds event embeddings (95 vectors, one per event)
  2. Builds the cosine similarity index
  3. Saves everything to data/processed/similarity/
  4. Runs demo queries to test similar fault clustering
  5. Computes intra-class vs inter-class similarity
  6. Saves a report to outputs/reports/similarity_index_results.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.similarity_index import (
    build_event_embeddings,
    build_similarity_index,
    find_similar_events,
    save_index,
)


def run_demo_queries(event_embeddings_df, index, n_queries=3, k=3):
    """
    Pick anomaly events and find their most similar events.
    Returns list of query results for the report.
    """
    emb_cols = [c for c in event_embeddings_df.columns if c.startswith("emb_")]

    # Select anomaly events from different farms for diverse demos
    anomaly_df = event_embeddings_df[event_embeddings_df["event_label"] == "anomaly"]

    # Try to get one from each farm for variety
    demo_events = []
    for farm in ["A", "B", "C"]:
        farm_anomalies = anomaly_df[anomaly_df["farm"] == farm]
        if len(farm_anomalies) > 0:
            demo_events.append(farm_anomalies.iloc[0])
            if len(demo_events) >= n_queries:
                break

    # Fill remaining slots if needed
    if len(demo_events) < n_queries:
        remaining = anomaly_df[
            ~anomaly_df.index.isin([e.name for e in demo_events])
        ]
        for _, row in remaining.iterrows():
            demo_events.append(row)
            if len(demo_events) >= n_queries:
                break

    query_results = []
    print("\n" + "=" * 70)
    print("DEMO: Similar Fault Finder")
    print("=" * 70)

    for event_row in demo_events:
        query_emb = event_row[emb_cols].values.astype(np.float32)
        farm = event_row["farm"]
        event_id = int(event_row["event_id"])
        desc = event_row["event_description"]
        label = event_row["event_label"]

        results = find_similar_events(
            query_embedding=query_emb,
            index=index,
            event_embeddings_df=event_embeddings_df,
            k=k,
            exclude_event=(farm, event_id),
        )

        print(f"\nQuery: Event {event_id} (Farm {farm}) - {desc} [{label}]")
        print(f"  Most similar events:")
        similar_strs = []
        for i, r in enumerate(results, 1):
            sim_pct = r["similarity_score"] * 100
            print(f"    {i}. Event {r['event_id']} (Farm {r['farm']}) - "
                  f"{r['event_description']} [{r['event_label']}] "
                  f"({sim_pct:.1f}% similar)")
            similar_strs.append(
                f"Event {r['event_id']} ({r['event_description']}, "
                f"{sim_pct:.1f}% similar)"
            )

        query_results.append({
            "query_farm": farm,
            "query_event_id": event_id,
            "query_description": desc,
            "query_label": label,
            "similar_events": results,
        })

    return query_results


def compute_class_similarities(event_embeddings_df, index):
    """
    Compute average intra-class vs inter-class similarity.

    Intra-class: anomaly-to-anomaly similarity
    Inter-class: anomaly-to-normal similarity
    """
    emb_cols = [c for c in event_embeddings_df.columns if c.startswith("emb_")]
    embeddings = event_embeddings_df[emb_cols].values.astype(np.float32)
    labels = event_embeddings_df["event_label"].values

    anomaly_mask = labels == "anomaly"
    normal_mask = labels == "normal"

    anomaly_embs = embeddings[anomaly_mask]
    normal_embs = embeddings[normal_mask]

    # Compute pairwise cosine similarity within anomalies
    intra_sims = []
    for i in range(len(anomaly_embs)):
        for j in range(i + 1, len(anomaly_embs)):
            # cosine similarity
            a, b = anomaly_embs[i], anomaly_embs[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(a, b) / (norm_a * norm_b)
                intra_sims.append(float(sim))

    # Compute pairwise cosine similarity between anomalies and normals
    inter_sims = []
    for i in range(len(anomaly_embs)):
        for j in range(len(normal_embs)):
            a, b = anomaly_embs[i], normal_embs[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(a, b) / (norm_a * norm_b)
                inter_sims.append(float(sim))

    # Compute within-normal similarity too for reference
    normal_sims = []
    for i in range(len(normal_embs)):
        for j in range(i + 1, len(normal_embs)):
            a, b = normal_embs[i], normal_embs[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(a, b) / (norm_a * norm_b)
                normal_sims.append(float(sim))

    avg_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    avg_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    avg_normal = float(np.mean(normal_sims)) if normal_sims else 0.0

    return {
        "anomaly_to_anomaly_avg_similarity": round(avg_intra, 4),
        "anomaly_to_normal_avg_similarity": round(avg_inter, 4),
        "normal_to_normal_avg_similarity": round(avg_normal, 4),
        "n_anomaly_pairs": len(intra_sims),
        "n_inter_pairs": len(inter_sims),
        "n_normal_pairs": len(normal_sims),
        "separation_ratio": round(avg_intra / avg_inter, 4) if avg_inter > 0 else None,
    }


def main():
    print("=" * 70)
    print("Phase 5.7: Building Vector Similarity Index")
    print("=" * 70)

    # Step 1: Build event embeddings
    print("\n[1/4] Building event embeddings (mean prediction-window vectors)...")
    event_embeddings_df = build_event_embeddings(str(PROJECT_ROOT))

    n_events = len(event_embeddings_df)
    emb_cols = [c for c in event_embeddings_df.columns if c.startswith("emb_")]
    n_dims = len(emb_cols)

    print(f"  Total events indexed: {n_events}")
    print(f"  Embedding dimensions: {n_dims}")
    print(f"  Events per farm:")
    for farm in ["A", "B", "C"]:
        farm_df = event_embeddings_df[event_embeddings_df["farm"] == farm]
        n_anom = (farm_df["event_label"] == "anomaly").sum()
        n_norm = (farm_df["event_label"] == "normal").sum()
        print(f"    Farm {farm}: {len(farm_df)} events "
              f"({n_anom} anomaly, {n_norm} normal)")

    # Step 2: Build similarity index
    print("\n[2/4] Building cosine similarity index...")
    index, embeddings_matrix = build_similarity_index(event_embeddings_df)
    print(f"  Index fitted on {embeddings_matrix.shape[0]} vectors "
          f"of {embeddings_matrix.shape[1]} dimensions")

    # Step 3: Save index
    save_dir = PROJECT_ROOT / "data" / "processed" / "similarity"
    print(f"\n[3/4] Saving index to {save_dir}...")
    save_index(event_embeddings_df, index, str(save_dir))

    # Step 4: Demo queries + analysis
    print("\n[4/4] Running demo queries and class similarity analysis...")

    # Demo queries
    query_results = run_demo_queries(event_embeddings_df, index, n_queries=3, k=3)

    # Class similarity analysis
    class_sims = compute_class_similarities(event_embeddings_df, index)

    print("\n" + "=" * 70)
    print("CLASS SIMILARITY ANALYSIS")
    print("=" * 70)
    print(f"  Anomaly-to-Anomaly avg similarity: "
          f"{class_sims['anomaly_to_anomaly_avg_similarity']:.4f}")
    print(f"  Anomaly-to-Normal avg similarity:   "
          f"{class_sims['anomaly_to_normal_avg_similarity']:.4f}")
    print(f"  Normal-to-Normal avg similarity:     "
          f"{class_sims['normal_to_normal_avg_similarity']:.4f}")
    if class_sims["separation_ratio"] is not None:
        print(f"  Separation ratio (intra/inter):     "
              f"{class_sims['separation_ratio']:.4f}")
        if class_sims["separation_ratio"] > 1.0:
            print("  --> Anomalies are more similar to each other than to normals "
                  "(good clustering!)")
        else:
            print("  --> No clear separation between anomaly and normal embeddings")

    # Save report
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "similarity_index_results.json"

    report = {
        "total_events_indexed": n_events,
        "embedding_dimensions": n_dims,
        "events_per_farm": {
            farm: int((event_embeddings_df["farm"] == farm).sum())
            for farm in ["A", "B", "C"]
        },
        "label_distribution": {
            "anomaly": int((event_embeddings_df["event_label"] == "anomaly").sum()),
            "normal": int((event_embeddings_df["event_label"] == "normal").sum()),
        },
        "demo_query_results": query_results,
        "class_similarity_analysis": class_sims,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("Phase 5.7 complete. Similarity index ready for dashboard.")
    print("=" * 70)


if __name__ == "__main__":
    main()
