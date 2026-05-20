import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


jsonl_path = "/home/wjh/Wan2.2/outputs/isolated_layer_error/isolated_layer_error.jsonl"
out_dir = Path("layer_error_plots")
out_dir.mkdir(parents=True, exist_ok=True)


def parse_module_name(name):
    """
    解析:
      blocks.0.cross_attn.k
      blocks.12.self_attn.o
      blocks.3.ffn.2
    """
    m = re.search(r"blocks\.(\d+)\.(self_attn|cross_attn)\.(q|k|v|o)", name)
    if m:
        block_id = int(m.group(1))
        module_type = f"{m.group(2)}.{m.group(3)}"
        return block_id, module_type

    m = re.search(r"blocks\.(\d+)\.ffn\.(0|2)", name)
    if m:
        block_id = int(m.group(1))
        module_type = f"ffn.{m.group(2)}"
        return block_id, module_type

    return None, "other"


rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        block_id, module_type = parse_module_name(row["module"])
        row["block_id"] = block_id
        row["module_type"] = module_type

        eps = 1e-12
        row["mae_over_mean_ref_abs"] = row["mae"] / max(row["mean_ref_abs"], eps)
        row["rmse_over_mean_ref_abs"] = row["rmse"] / max(row["mean_ref_abs"], eps)
        row["max_err_over_ref_absmax"] = row["max_abs_error"] / max(row["ref_absmax"], eps)
        row["log10_cosine_error"] = np.log10(max(row["cosine_error"], eps))
        rows.append(row)

df = pd.DataFrame(rows)

# 排序，方便后续画图
df = df.sort_values(["expert", "block_id", "module_type", "module"]).reset_index(drop=True)

# 保存带派生字段的 CSV
df.to_csv(out_dir / "layer_error_with_derived_metrics.csv", index=False)

print("Loaded rows:", len(df))
print("\nTop-20 by rel_l2:")
print(df.sort_values("rel_l2", ascending=False)[
    ["expert", "module", "rel_l2", "cosine", "cosine_error", "mae", "rmse", "max_abs_error"]
].head(20).to_string(index=False))

print("\nTop-20 by cosine_error:")
print(df.sort_values("cosine_error", ascending=False)[
    ["expert", "module", "rel_l2", "cosine", "cosine_error"]
].head(20).to_string(index=False))

print("\nTop-20 by max_err_over_ref_absmax:")
print(df.sort_values("max_err_over_ref_absmax", ascending=False)[
    ["expert", "module", "max_abs_error", "ref_absmax", "max_err_over_ref_absmax", "rel_l2"]
].head(20).to_string(index=False))


# 1. rel_l2 按层顺序折线图
for expert, sub in df.groupby("expert"):
    sub = sub.reset_index(drop=True)

    plt.figure(figsize=(18, 5))
    plt.plot(np.arange(len(sub)), sub["rel_l2"].values, marker=".", linewidth=1)
    plt.xlabel("Layer index")
    plt.ylabel("rel_l2")
    plt.title(f"{expert} isolated layer error: rel_l2 by layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{expert}_rel_l2_by_layer.png", dpi=200)
    plt.close()


# 2. block_id × module_type heatmap
module_order = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]

def plot_heatmap(metric):
    for expert, sub in df.groupby("expert"):
        pivot = sub.pivot_table(
            index="block_id",
            columns="module_type",
            values=metric,
            aggfunc="mean",
        )

        pivot = pivot.reindex(columns=[c for c in module_order if c in pivot.columns])

        plt.figure(figsize=(14, 8))
        im = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(im, label=metric)
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel("module_type")
        plt.ylabel("block_id")
        plt.title(f"{expert}: {metric} heatmap")
        plt.tight_layout()
        plt.savefig(out_dir / f"{expert}_{metric}_heatmap.png", dpi=200)
        plt.close()

plot_heatmap("rel_l2")
plot_heatmap("cosine_error")
plot_heatmap("max_err_over_ref_absmax")


# 3. Top-30 rel_l2 bar chart
for expert, sub in df.groupby("expert"):
    top = sub.sort_values("rel_l2", ascending=False).head(30).iloc[::-1]

    plt.figure(figsize=(12, 10))
    plt.barh(top["module"], top["rel_l2"])
    plt.xlabel("rel_l2")
    plt.title(f"{expert}: Top-30 worst layers by rel_l2")
    plt.tight_layout()
    plt.savefig(out_dir / f"{expert}_top30_rel_l2.png", dpi=200)
    plt.close()


# 4. rel_l2 vs cosine_error scatter
for expert, sub in df.groupby("expert"):
    plt.figure(figsize=(8, 6))
    plt.scatter(sub["rel_l2"], sub["cosine_error"], s=20)
    plt.xlabel("rel_l2")
    plt.ylabel("cosine_error")
    plt.title(f"{expert}: rel_l2 vs cosine_error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{expert}_rel_l2_vs_cosine_error.png", dpi=200)
    plt.close()


# 5. 按 module_type 聚合
summary = df.groupby(["expert", "module_type"]).agg(
    rel_l2_mean=("rel_l2", "mean"),
    rel_l2_max=("rel_l2", "max"),
    cosine_error_mean=("cosine_error", "mean"),
    cosine_error_max=("cosine_error", "max"),
    max_err_ratio_mean=("max_err_over_ref_absmax", "mean"),
    max_err_ratio_max=("max_err_over_ref_absmax", "max"),
    num_layers=("module", "count"),
).reset_index()

summary.to_csv(out_dir / "summary_by_module_type.csv", index=False)

print("\nSummary by module_type:")
print(summary.sort_values(["expert", "rel_l2_mean"], ascending=[True, False]).to_string(index=False))

print(f"\nSaved plots and CSV to: {out_dir}")