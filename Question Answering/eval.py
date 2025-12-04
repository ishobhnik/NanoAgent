import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Tuple, Any
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """You are a financial analyst.
Context:
{doc_text}

Question: {question}

Instructions:
- Answer the question based on the Context.
- Output ONLY the requested number or phrase.
- Do NOT explain your reasoning or show calculations.
- If the answer is a percentage, include the % sign.

Answer:"""

def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    
    print(f"Loading data from: {p}")
    if not p.exists():
        alt = (Path(__file__).parent / path).resolve()
        if alt.exists(): p = alt
        else: raise FileNotFoundError(f"Dataset not found at: {p}")

    ext = p.suffix.lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    elif ext == ".jsonl":
        df = pd.read_json(p, lines=True)
    elif ext == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported dataset extension: {ext}")

    if "id" not in df.columns:
        df["id"] = range(len(df))
    return df

def build_prompt(row: pd.Series) -> Tuple[str, str]:
    def get(names):
        for n in names:
            for c in row.index:
                if c.lower() == n.lower(): return row[c]
        return None

    qa_obj = get(["qa", "qa_pair"])
    if isinstance(qa_obj, dict):
        q_str = str(qa_obj.get("question", "") or qa_obj.get("query", ""))
        gold_str = str(qa_obj.get("answer", ""))
    else:
        q_str = str(get(["question", "query"]) or "")
        gold_str = str(get(["answer", "gold", "label", "output"]) or "")

    pre = get(["pre_text"])
    post = get(["post_text"])
    full_text = get(["text", "context", "passage"])

    doc_text = ""
    if full_text:
        doc_text = str(full_text)
    else:
        if isinstance(pre, list): pre = " ".join(str(x) for x in pre)
        if isinstance(post, list): post = " ".join(str(x) for x in post)
        doc_text = f"{pre or ''}\n{post or ''}".strip()

    table_raw = get(["table", "table_text", "table_ori"])
    table_text = ""
    if table_raw:
        if isinstance(table_raw, list):
            table_text = "\n".join([" | ".join(map(str, r)) for r in table_raw])
        else:
            table_text = str(table_raw)
            
    if table_text:
        doc_text += f"\n\nTable Data:\n{table_text}"

    if len(doc_text) > 12000: 
        doc_text = doc_text[:12000] + "...(truncated)"

    prompt = PROMPT_TEMPLATE.format(doc_text=doc_text, question=q_str)
    return prompt, gold_str

def extract_number(text):
    matches = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?', text)
    if not matches: return None
    return matches[-1]

def parse_val(s):
    if not s: return None
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    is_pct = "%" in s
    s = s.replace("%", "")
    try:
        val = float(s)
        return val, is_pct
    except:
        return None

def em_match(pred, gold):
    if str(pred).strip().lower() == str(gold).strip().lower():
        return True

    pred_num_str = extract_number(pred)
    gold_num_str = extract_number(gold)

    if pred_num_str and gold_num_str:
        p_val = parse_val(pred_num_str)
        g_val = parse_val(gold_num_str)
        
        if p_val and g_val:
            pv, p_pct = p_val
            gv, g_pct = g_val
            
            if p_pct and not g_pct: pv = pv / 100
            if g_pct and not p_pct: gv = gv / 100
            
            if abs(pv - gv) < 1e-4: return True
            if gv != 0 and abs((pv - gv) / gv) < 0.01: return True

    return False

def load_model(model_id):
    print(f"Loading Model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware: {device.upper()}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    return tok, model

def generate(model, tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        text_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        text_input = f"User: {prompt}\nAssistant:"

    inputs = tok(text_input, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  
            do_sample=False,   
            pad_token_id=tok.eos_token_id
        )

    return tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="full_results_direct.csv")
    ap.add_argument("--model_id", default="google/gemma-2b-it")
    ap.add_argument("--limit", type=int, default=None) 
    args = ap.parse_args()

    df = load_dataset(args.data)
    tok, model = load_model(args.model_id)
    
    n_total = len(df)
    n = min(args.limit, n_total) if args.limit else n_total
    
    print(f"\nSTARTING FULL RUN (Direct Answer Mode): {n} rows")
    print("-" * 30)

    records = []
    correct_count = 0
    start_time = time.time()

    for i in range(n):
        try:
            row = df.iloc[i]
            prompt, gold = build_prompt(row)
            pred = generate(model, tok, prompt)
            
            is_correct = em_match(pred, gold)
            if is_correct: correct_count += 1
            
            records.append({
                "id": row.get("id", i),
                "gold": gold,
                "pred": pred,
                "match": is_correct
            })
            
            if (i + 1) % 100 == 0 or (i + 1) == n:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                print(f"[{i+1}/{n}] Acc: {correct_count/(i+1):.2%} | {avg_time:.2f}s/row")

        except KeyboardInterrupt:
            print("\nRun interrupted by user. Saving progress...")
            break
        except Exception as e:
            print(f"Error row {i}: {e}")
            continue

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    
    print("-" * 30)
    print(f"Final Accuracy: {correct_count/len(records):.2%}")
    print(f"Results saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()