import os
import time
import requests
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

project_dir = Path("./documerica_project")
image_dir = project_dir / "images"
output_dir = project_dir / "outputs"
image_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

meta_df = pd.read_excel("documerica_images_cleaned.xlsx")
print(f"Loaded {len(meta_df)} records")

print("\nDownloading images...")
downloads = []
for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    img_name = str(row["image_name"])
    if not Path(img_name).suffix:
        img_name += ".jpg"
    
    out_path = image_dir / img_name
    
    try:
        r = requests.get(row["image_url"], timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        out_path.write_bytes(r.content)
        downloads.append({"image_name": img_name, "image_id": row["image_id"], "status": "success"})
    except Exception as e:
        downloads.append({"image_name": img_name, "image_id": row["image_id"], "status": "failed"})
    
    time.sleep(0.1)

dl_df = pd.DataFrame(downloads)
dl_df.to_csv(output_dir / "downloads.csv", index=False)

ok = (dl_df["status"] == "success").sum()
bad = (dl_df["status"] == "failed").sum()
print(f"OK: {ok}, Failed: {bad}")

print("\nLoading model...")
torch.cuda.empty_cache()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model.eval()
print(f"Device: {model.device}")

prompt = """You are analyzing a historical cultural heritage photograph.

Describe and explain the image based ONLY on what is visually observable.
Separate direct visual observations from interpretations or inferences.

Use cautious language such as "appears to be", "may be", or "could indicate" 
whenever something cannot be established directly from the image.

Discuss relevant:
- people and their visible actions
- objects and structures
- setting and environment
- activities
- visible text or signs
- possible contextual clues

Do NOT invent or confidently assert:
- dates, locations, names, occupations, historical events, cultural identities
- relationships, photographer information
- object identities that are visually uncertain
- or other facts that cannot be established from the image

If something cannot be determined, explicitly state that.

Produce approximately 120-180 words."""

seeds = [1001, 1002, 1003, 1004, 1005]

def gen_explanation(img_path, seed):
    torch.cuda.empty_cache()
    
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[txt], images=[img], padding=True, return_tensors="pt")
    inp = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inp.items()}
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    with torch.inference_mode():
        out_ids = model.generate(**inp, max_new_tokens=120, temperature=0.8, top_p=0.9, do_sample=True, use_cache=True)
    
    trimmed = [o[len(i):] for i, o in zip(inp["input_ids"], out_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    del inp, out_ids, trimmed
    torch.cuda.empty_cache()
    return text.strip()

ckpt_file = output_dir / "ckpt.csv"
if ckpt_file.exists():
    results = pd.read_csv(ckpt_file)
    print(f"\nResuming: {len(results)} done")
else:
    results = pd.DataFrame()

ok_images = dl_df[dl_df["status"] == "success"]
print(f"Generating {len(ok_images)} images × {len(seeds)} seeds = {len(ok_images) * len(seeds)} total")

print("\nGenerating...")
for _, row in tqdm(ok_images.iterrows(), total=len(ok_images)):
    img_path = image_dir / row["image_name"]
    
    for exp_num, seed in enumerate(seeds, start=1):
        done = ((results["image_name"] == row["image_name"]) & (results["exp_num"] == exp_num)).any()
        if done:
            continue
        
        try:
            exp = gen_explanation(img_path, seed=seed)
        except Exception as e:
            print(f"Error {row['image_name']} ({exp_num}): {e}")
            continue
        
        new = pd.DataFrame([{
            "image_name": row["image_name"],
            "image_id": row["image_id"],
            "exp_num": exp_num,
            "seed": seed,
            "explanation": exp,
        }])
        results = pd.concat([results, new], ignore_index=True)
        results.to_csv(ckpt_file, index=False)

print(f"\nGenerated {len(results)}")

final = results.drop_duplicates(subset=["image_name", "exp_num"]).sort_values(["image_name", "exp_num"]).reset_index(drop=True)
final.to_csv(output_dir / "explanations.csv", index=False, quoting=1)
print(f"Saved to {output_dir / 'explanations.csv'}")
print(f"Images: {final['image_name'].nunique()}")

