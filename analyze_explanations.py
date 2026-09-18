import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path
from itertools import combinations
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

out_dir = Path("./documerica_project/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading explanations...")
df = pd.read_csv(out_dir / "explanations.csv")
print(f"Loaded {len(df)} from {df['image_name'].nunique()} images")

print("\nSegmenting sentences...")
sent_rows = []
for idx, row in df.iterrows():
    for snum, sent in enumerate(sent_tokenize(str(row["explanation"])), start=1):
        sent_rows.append({
            "image_name": row["image_name"],
            "exp_idx": idx,
            "sent_num": snum,
            "sent": sent,
        })

sent_df = pd.DataFrame(sent_rows).reset_index(drop=True)
sent_df["sent_id"] = sent_df.index
print(f"Total: {len(sent_df)} sentences")
sent_df.to_csv(out_dir / "sentences.csv", index=False)

print("\nEmbedding sentences...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
embeds = embed_model.encode(sent_df["sent"].tolist(), batch_size=32, show_progress_bar=True, normalize_embeddings=True)
np.save(out_dir / "embeddings.npy", embeds)
print(f"Shape: {embeds.shape}")

embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

print("\nPairing sentences...")
pair_rows = []
for img_name, grp in sent_df.groupby("image_name"):
    recs = grp.to_dict("records")
    for a, b in combinations(recs, 2):
        if a["exp_idx"] == b["exp_idx"]:
            continue
        pair_rows.append({
            "image_name": img_name,
            "src_id": a["sent_id"],
            "tgt_id": b["sent_id"],
            "src_sent": a["sent"],
            "tgt_sent": b["sent"],
        })

pairs_df = pd.DataFrame(pair_rows)
src_vecs = embeds[pairs_df["src_id"].to_numpy()]
tgt_vecs = embeds[pairs_df["tgt_id"].to_numpy()]
pairs_df["sim"] = np.sum(src_vecs * tgt_vecs, axis=1)

print(f"Total pairs: {len(pairs_df)}")
print(f"Mean sim: {pairs_df['sim'].mean():.3f}")

for t in [0.50, 0.60, 0.70, 0.80]:
    c = (pairs_df["sim"] >= t).sum()
    print(f"  {t}: {c}")

thresh = 0.60
cand_pairs = pairs_df[pairs_df["sim"] >= thresh].reset_index(drop=True)
print(f"\nCandidates (>= {thresh}): {len(cand_pairs)}")

print("\nRunning NLI...")
nli_pipe = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base", device=0 if torch.cuda.is_available() else -1, truncation=True)
nli_model = nli_pipe.model
nli_model.eval()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model.to(dev)

nli_res = []
bs = 16
for start in range(0, len(cand_pairs), bs):
    batch = cand_pairs.iloc[start:start + bs]
    
    inp = nli_pipe.tokenizer(
        batch["src_sent"].astype(str).tolist(),
        batch["tgt_sent"].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inp = {k: v.to(dev) for k, v in inp.items()}
    
    with torch.no_grad():
        logits = nli_model(**inp).logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    for j in range(len(batch)):
        row = batch.iloc[j]
        lid = preds[j].item()
        nli_res.append({
            "image_name": row["image_name"],
            "src_id": row["src_id"],
            "tgt_id": row["tgt_id"],
            "src_sent": row["src_sent"],
            "tgt_sent": row["tgt_sent"],
            "sim": row["sim"],
            "label": nli_model.config.id2label[lid],
            "score": probs[j, lid].item(),
        })

nli_df = pd.DataFrame(nli_res)
nli_df.to_csv(out_dir / "nli.csv", index=False)
print(f"NLI:\n{nli_df['label'].value_counts()}")

print("\nImage-level consistency...")
def img_summary(grp):
    e = (grp["label"] == "entailment").sum()
    c = (grp["label"] == "contradiction").sum()
    n = (grp["label"] == "neutral").sum()
    d = e + c
    return pd.Series({
        "pairs": len(grp),
        "ent": e,
        "con": c,
        "neu": n,
        "con_rate": c / len(grp),
        "cons_score": e / d if d > 0 else np.nan,
    })

img_sum = nli_df.groupby("image_name").apply(img_summary, include_groups=False).reset_index()
img_sum["cons_pct"] = img_sum["cons_score"] * 100
img_sum.to_csv(out_dir / "image_summary.csv", index=False)

print(f"Mean con rate: {(img_sum['con_rate'] * 100).mean():.2f}%")
print(f"Mean cons: {img_sum['cons_pct'].mean():.2f}%")

plt.figure(figsize=(14, 6))
plot = img_sum.sort_values("con_rate", ascending=False)
plt.bar(plot["image_name"], plot["con_rate"] * 100)
plt.xlabel("Image")
plt.ylabel("Contradiction rate (%)")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig(out_dir / "con_rates.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nClustering...")
def cluster_sents(indices, t):
    reps = []
    assign = {}
    for idx in indices:
        v = embeds[idx]
        cid = None
        for c, rid in enumerate(reps):
            if np.dot(v, embeds[rid]) >= t:
                cid = c
                break
        if cid is None:
            reps.append(idx)
            cid = len(reps) - 1
        assign[idx] = cid
    return assign

cthresh = 0.75
claim_rows = []
for img_name, grp in sent_df.groupby("image_name"):
    assign = cluster_sents(grp["sent_id"].tolist(), cthresh)
    for _, row in grp.iterrows():
        claim_rows.append({
            "image_name": img_name,
            "sent_id": row["sent_id"],
            "claim_id": f"{img_name}_c{assign[row['sent_id']]}",
            "sent": row["sent"],
        })

claims = pd.DataFrame(claim_rows)
print(f"Claims: {claims['claim_id'].nunique()}")
claims.to_csv(out_dir / "claims.csv", index=False)

print("\nMatching metadata...")
meta = pd.read_excel("documerica_images_cleaned.xlsx")

def strip_ext(s):
    return re.sub(r"\.[^.]+$", "", str(s).strip())

claims["img_key"] = claims["image_name"].apply(strip_ext)
meta["img_key"] = meta["image_name"].apply(strip_ext)

claim_meta = claims.merge(meta, on="img_key", how="left", suffixes=("_c", "_m"), validate="many_to_one")

claim_lvl = (
    claim_meta.groupby("claim_id", as_index=False)
    .agg({
        "img_key": "first",
        "image_name_c": "first",
        "sent": lambda s: " ".join(dict.fromkeys(str(x) for x in s)),
        "city": "first",
        "state": "first",
        "date_taken": "first",
        "keywords": "first",
        "exhibit_description": "first",
    })
    .rename(columns={"image_name_c": "image_name", "sent": "claim_txt"})
)

for col in ["city", "state", "date_taken", "keywords", "exhibit_description"]:
    claim_lvl[col] = claim_lvl[col].fillna("").astype(str).str.strip()

print(f"With meta: {claim_lvl['img_key'].notna().sum()}")

print("\nClassifying fields...")
date_words = ["date", "dated", "year", "1960", "1970", "1980", "1990", "2000"]
loc_words = ["location", "located", "city", "state", "street", "road", "town", "building"]
ctx_words = ["government", "epa", "environmental", "industrial", "residential", "institution"]

def get_field(t):
    t = str(t).lower()
    if any(w in t for w in date_words):
        return "date"
    if any(w in t for w in loc_words):
        return "location"
    if any(w in t for w in ctx_words):
        return "context"
    return "visual"

claim_lvl["field"] = claim_lvl["claim_txt"].apply(get_field)
print(f"Fields:\n{claim_lvl['field'].value_counts()}")

print("\nAssessing metadata support...")
def assess_meta(row):
    c = row["claim_txt"].lower()
    f = row["field"]
    city, state, dt = row["city"].lower(), row["state"].lower(), row["date_taken"].lower()
    kw = row["keywords"].lower()
    
    if f == "visual":
        return "not_checkable"
    if f == "location":
        cands = [v for v in (city, state) if v]
        if not cands:
            return "not_checkable"
        return "supported" if any(v in c for v in cands) else "unverifiable"
    if f == "date":
        if not dt:
            return "not_checkable"
        cy = set(re.findall(r"\b(?:19|20)\d{2}\b", c))
        my = set(re.findall(r"\b(?:19|20)\d{2}\b", dt))
        if not cy or not my:
            return "unverifiable"
        return "supported" if cy & my else "unsupported"
    if f == "context":
        if not kw:
            return "not_checkable"
        cw = set(re.findall(r"\b[a-z]{4,}\b", c))
        mw = set(re.findall(r"\b[a-z]{4,}\b", kw))
        return "supported" if cw & mw else "unverifiable"
    return "unverifiable"

claim_lvl["meta_support"] = claim_lvl.apply(assess_meta, axis=1)
claim_lvl.to_csv(out_dir / "claims_meta.csv", index=False)
print(f"Support:\n{claim_lvl['meta_support'].value_counts()}")

print("\nReliability...")
rel_df = img_sum.copy()
rel_df["rel_score"] = rel_df["cons_pct"]

def rel_cat(s):
    if s < 60:
        return "low"
    elif s < 80:
        return "moderate"
    return "high"

rel_df["cat"] = rel_df["rel_score"].apply(rel_cat)
rel_df = rel_df.sort_values("rel_score").reset_index(drop=True)
rel_df.to_csv(out_dir / "reliability.csv", index=False)

print(f"Category:\n{rel_df['cat'].value_counts()}")

max_diff = (rel_df["rel_score"] - rel_df["cons_pct"]).abs().max()
print(f"\nVerify: max |rel - cons| = {max_diff:.2e}")
assert max_diff < 1e-9, "Not pure consistency!"
print("✓ Reliability is pure NLI consistency")

low = rel_df.loc[rel_df["rel_score"].idxmin()]
print(f"Min: {low['image_name']} = {low['rel_score']:.2f}%")

plt.figure(figsize=(14, 6))
p = rel_df.sort_values("rel_score")
plt.plot(range(len(p)), p["rel_score"], marker="o", markersize=3)
plt.axhline(60, linestyle="--", alpha=0.5)
plt.axhline(80, linestyle="--", alpha=0.5)
plt.xlabel("Images")
plt.ylabel("Reliability (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "reliability.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nWeight sensitivity...")
meta_by_img = (
    claim_lvl.groupby("image_name")["meta_support"]
    .agg(sup=lambda s: (s == "supported").sum(), unsup=lambda s: (s == "unsupported").sum())
)
meta_by_img["checkable"] = meta_by_img["sup"] + meta_by_img["unsup"]
meta_by_img["sup_score"] = np.where(meta_by_img["checkable"] > 0, meta_by_img["sup"] / meta_by_img["checkable"], np.nan)

sens = rel_df.merge(meta_by_img[["sup_score"]], on="image_name", how="left")
pri = sens["rel_score"]

for nw, mw in [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]:
    scores = np.where(
        sens["sup_score"].notna(),
        (nw * sens["cons_score"] + mw * sens["sup_score"]) * 100,
        sens["cons_score"] * 100,
    )
    rho, p = spearmanr(pri, scores)
    print(f"  NLI {nw:.0%} + Meta {mw:.0%}: ρ={rho:.4f}")

print(f"\nDone. Output: {out_dir}")

