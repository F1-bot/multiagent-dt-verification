"""Perception agent on CrisisMMD v2.0 (damage-severity task).

Steps: CLIP ViT-L/14 image/text embeddings -> linear heads (train split) -> late fusion + temperature
scaling (dev split) -> Grad-CAM on the image head (tokens entering the last ViT block) with a deletion
faithfulness test -> tweet sentiment (TimeLMs RoBERTa) -> image difference-hash duplicates.
Usage: python perception_features.py <crisismmd_dir> <out_dir> [features_csv]
"""
import os
import sys
import json
import time

# Hugging Face models are cached in the default location (set HF_HOME to change it).
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import minimize_scalar

sys.stdout.reconfigure(encoding="utf-8")
DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
LABELS = ["little_or_no_damage", "mild_damage", "severe_damage"]

rows = []
for s in ["train", "dev", "test"]:
    with open(os.path.join(DATA, f"damage_{s}.json"), encoding="utf-8") as f:
        for d in json.load(f):
            d["split"] = s
            rows.append(d)
df = pd.DataFrame(rows)
df["y"] = df["label"].map({l: i for i, l in enumerate(LABELS)})
print(df.groupby(["split", "label"]).size(), flush=True)

# ---------------- CLIP embeddings ----------------
t0 = time.time()
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def load_img(p):
    return Image.open(os.path.join(DATA, "images", p)).convert("RGB")


def as_tensor(x):
    if torch.is_tensor(x):
        return x
    return getattr(x, "pooler_output", None) if getattr(x, "pooler_output", None) is not None else x[0]


emb_path = os.path.join(OUT, "clip_embeddings.npz")
if os.path.exists(emb_path):
    z = np.load(emb_path)
    img_emb, txt_emb = z["img"], z["txt"]
    print("loaded cached embeddings", img_emb.shape, flush=True)
else:
    img_emb, txt_emb = [], []
    BS = 32
    with torch.no_grad():
        for i in range(0, len(df), BS):
            sub = df.iloc[i:i + BS]
            pv = proc(images=[load_img(p) for p in sub.image_path], return_tensors="pt")["pixel_values"].to(DEV)
            e = as_tensor(clip.get_image_features(pixel_values=pv))
            img_emb.append(torch.nn.functional.normalize(e, dim=-1).cpu())
            tk = proc.tokenizer(list(sub.tweet_text), padding=True, truncation=True, max_length=77,
                                return_tensors="pt").to(DEV)
            t = as_tensor(clip.get_text_features(**tk))
            txt_emb.append(torch.nn.functional.normalize(t, dim=-1).cpu())
    img_emb = torch.cat(img_emb).numpy()
    txt_emb = torch.cat(txt_emb).numpy()
df["kappa"] = (img_emb * txt_emb).sum(1)
np.savez_compressed(os.path.join(OUT, "clip_embeddings.npz"), img=img_emb, txt=txt_emb)
print(f"embeddings done {time.time() - t0:.0f}s", flush=True)

# ---------------- linear heads (train), model selection + calibration (dev) ----------------
tr = (df.split == "train").values
dv = (df.split == "dev").values
te = (df.split == "test").values
y = df.y.values
SCALE = 10.0


def fit_head(X):
    best = None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]:
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
        ll = -np.mean(np.log(m.predict_proba(X[dv])[np.arange(dv.sum()), y[dv]] + 1e-12))
        if best is None or ll < best[0]:
            best = (ll, C, m)
    print(f"  best C={best[1]} dev NLL={best[0]:.4f}", flush=True)
    return best[2]


h_img = fit_head(img_emb * SCALE)
h_txt = fit_head(txt_emb * SCALE)
L_img = h_img.decision_function(img_emb * SCALE)
L_txt = h_txt.decision_function(txt_emb * SCALE)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(L, idx):
    P = softmax(L[idx])
    return -np.mean(np.log(P[np.arange(idx.sum()), y[idx]] + 1e-12))


beta = min((nll(L_img + b * L_txt, dv), b) for b in np.linspace(0, 1.5, 31))[1]
L_fus = L_img + beta * L_txt
T_img = minimize_scalar(lambda T: nll(L_img / T, dv), bounds=(0.05, 20), method="bounded").x
T_txt = minimize_scalar(lambda T: nll(L_txt / T, dv), bounds=(0.05, 20), method="bounded").x
T_fus = minimize_scalar(lambda T: nll(L_fus / T, dv), bounds=(0.05, 20), method="bounded").x
P_img, P_txt, P_fus = softmax(L_img / T_img), softmax(L_txt / T_txt), softmax(L_fus / T_fus)
print(f"beta={beta:.2f} T_img={T_img:.2f} T_txt={T_txt:.2f} T_fus={T_fus:.2f}", flush=True)
perf = {}
for nm, P in [("image", P_img), ("text", P_txt), ("fused", P_fus)]:
    pr = P.argmax(1)
    perf[nm] = {"acc": accuracy_score(y[te], pr[te]),
                "macroF1": f1_score(y[te], pr[te], average="macro"),
                "weightedF1": f1_score(y[te], pr[te], average="weighted")}
    print(f"  TEST {nm}: {perf[nm]}", flush=True)
for k in range(3):
    df[f"p_img{k}"], df[f"p_txt{k}"], df[f"p_fus{k}"] = P_img[:, k], P_txt[:, k], P_fus[:, k]


def norm_entropy(P):
    return -(P * np.log(P + 1e-12)).sum(1) / np.log(P.shape[1])


df["conf"] = 1 - norm_entropy(P_fus)
M = 0.5 * (P_img + P_txt)
js = 0.5 * (P_img * np.log((P_img + 1e-12) / (M + 1e-12))).sum(1) \
    + 0.5 * (P_txt * np.log((P_txt + 1e-12) / (M + 1e-12))).sum(1)
df["agree"] = 1 - js / np.log(2)
df["pred"] = P_fus.argmax(1)
df["correct"] = (df.pred == df.y).astype(int)

# ---------------- Grad-CAM on the image head + deletion faithfulness ----------------
W = torch.tensor(h_img.coef_ * SCALE / T_img, dtype=torch.float32, device=DEV)
b0 = torch.tensor(h_img.intercept_ / T_img, dtype=torch.float32, device=DEV)
store = {}


def hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.requires_grad:
        h.retain_grad()
        store["h"] = h


handle = clip.vision_model.encoder.layers[-2].register_forward_hook(hook)


def img_logits(pv):
    out = clip.vision_model(pixel_values=pv)
    e = torch.nn.functional.normalize(clip.visual_projection(out.pooler_output), dim=-1)
    return e @ W.T + b0


GRID, PATCH = 16, 14
K_TOP = int(0.2 * GRID * GRID)
rng = np.random.default_rng(0)
n = len(df)
cam_conc, top_share, del_drop, rand_drop = (np.full(n, np.nan) for _ in range(4))
idx_eval = np.where(~tr)[0]


def mask_patches(pv, indices):
    pm = pv.clone()
    for r in range(pv.shape[0]):
        for t in indices[r]:
            gy, gx = divmod(int(t), GRID)
            pm[r, :, gy * PATCH:(gy + 1) * PATCH, gx * PATCH:(gx + 1) * PATCH] = 0.0
    return pm


for s in range(0, len(idx_eval), 16):
    ii = idx_eval[s:s + 16]
    pv = proc(images=[load_img(p) for p in df.image_path.values[ii]], return_tensors="pt")["pixel_values"].to(DEV)
    clip.zero_grad(set_to_none=True)
    lg = img_logits(pv)
    pred = lg.argmax(1)
    lg[torch.arange(len(ii)), pred].sum().backward()
    A = store["h"][:, 1:, :].detach()
    G = store["h"].grad[:, 1:, :]
    cam = torch.relu((A * G.mean(1, keepdim=True)).sum(-1))
    cam = cam / (cam.sum(1, keepdim=True) + 1e-12)
    ent = -(cam * torch.log(cam + 1e-12)).sum(1) / np.log(GRID * GRID)
    top = cam.topk(K_TOP, dim=1)
    with torch.no_grad():
        p0 = torch.softmax(lg.detach(), 1)[torch.arange(len(ii)), pred]
        p_del = torch.softmax(img_logits(mask_patches(pv, top.indices.cpu().numpy())), 1)[torch.arange(len(ii)), pred]
        rnd = np.stack([rng.choice(GRID * GRID, K_TOP, replace=False) for _ in ii])
        p_rnd = torch.softmax(img_logits(mask_patches(pv, rnd)), 1)[torch.arange(len(ii)), pred]
    cam_conc[ii] = (1 - ent).cpu().numpy()
    top_share[ii] = top.values.sum(1).detach().cpu().numpy()
    del_drop[ii] = (p0 - p_del).cpu().numpy()
    rand_drop[ii] = (p0 - p_rnd).cpu().numpy()
    if s % 320 == 0:
        print(f"  gradcam {s}/{len(idx_eval)} {time.time() - t0:.0f}s", flush=True)
handle.remove()
df["cam_conc"], df["cam_top20"], df["del_drop"], df["rand_drop"] = cam_conc, top_share, del_drop, rand_drop
df["faith"] = df.del_drop - df.rand_drop
del clip
torch.cuda.empty_cache()

# ---------------- tweet sentiment ----------------
name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tok = AutoTokenizer.from_pretrained(name)
sm = AutoModelForSequenceClassification.from_pretrained(name).to(DEV).eval()


def prep(t):
    out = []
    for w in str(t).split(" "):
        if w.startswith("@") and len(w) > 1:
            w = "@user"
        elif w.startswith("http"):
            w = "http"
        out.append(w)
    return " ".join(out)


S = []
with torch.no_grad():
    for i in range(0, n, 64):
        x = tok([prep(t) for t in df.tweet_text.values[i:i + 64]], padding=True, truncation=True,
                max_length=128, return_tensors="pt").to(DEV)
        S.append(torch.softmax(sm(**x).logits, -1).cpu().numpy())
S = np.concatenate(S)
sent_labels = [sm.config.id2label[i].lower() for i in range(S.shape[1])]
for j, l in enumerate(sent_labels):
    df[f"sent_{l}"] = S[:, j]
df["sent_intensity"] = np.abs(df["sent_positive"] - df["sent_negative"])

# ---------------- image difference-hash near-duplicates ----------------


def dhash(p, size=8):
    im = Image.open(os.path.join(DATA, "images", p)).convert("L").resize((size + 1, size))
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


bits = np.stack([dhash(p) for p in df.image_path]).astype(np.uint8)
dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)
np.fill_diagonal(dist, 99)
df["dup"] = (dist.min(1) <= 4).astype(int)

FEAT_OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(OUT, "crisismmd_features.csv")
# tweet texts are not stored (privacy); tweet and image ids allow re-joining with CrisisMMD
df.drop(columns=["tweet_text", "image_path"]).to_csv(FEAT_OUT, index=False, encoding="utf-8")
with open(os.path.join(OUT, "perception_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"beta": float(beta), "T_img": float(T_img), "T_txt": float(T_txt), "T_fus": float(T_fus),
               "labels": LABELS, "sentiment_labels": sent_labels, "test_performance": perf,
               "n": {"train": int(tr.sum()), "dev": int(dv.sum()), "test": int(te.sum())}}, f, indent=2)
print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)
