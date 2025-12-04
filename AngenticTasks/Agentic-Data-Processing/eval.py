import os
import json
import glob
import re
import pandas as pd
import numpy as np
import warnings
import math
from scipy import stats
from typing import List, Dict

# --- HuggingFace & LangChain Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# --- Tools ---
try:
    from langchain_experimental.tools import PythonAstREPLTool
except ImportError:
    raise ImportError("Please install: pip install langchain-experimental")

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_NAME = "google/gemma-2b-it"
SANDBOX_DIR = "sandbox/artifacts"
NUM_RUNS = 8  # 'R' for statistics

# ------------------------------------------------------------------
# 1. FILE DISCOVERY & GROUND TRUTH GENERATION
# ------------------------------------------------------------------

def get_csv_files(base_dir: str) -> List[str]:
    """Recursively finds all .csv files in the artifact directories."""
    pattern = os.path.join(base_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"WARNING: No CSV files found in {base_dir}")
    return sorted(files)

def calculate_gold_standard(csv_path: str) -> Dict:
    """Reads the existing CSV to establish the 'Correct Answer'."""
    try:
        df = pd.read_csv(csv_path)
        
        # EXACT COLUMN MAPPING based on your input
        # C_ID, C_NAME, AGE_YRS, LOC_CD, REG_DT
        
        if "AGE_YRS" not in df.columns:
            # Fallback for case sensitivity
            df.columns = [c.upper() for c in df.columns]
        
        if "AGE_YRS" not in df.columns:
             return {"error": f"Column 'AGE_YRS' missing. Found: {list(df.columns)}"}

        return {
            "total_customers": len(df),
            "average_age": float(df["AGE_YRS"].mean())
        }
    except Exception as e:
        return {"error": str(e)}

# ------------------------------------------------------------------
# 2. AGENT SETUP
# ------------------------------------------------------------------

def load_agent(model_name: str):
    print(f"Loading Model: {model_name}...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. FORCE A GENERIC CHAT TEMPLATE
    # This fixes the "tokenizer.chat_template is not set" error.
    # It structures the prompt as "System: ... User: ... Assistant: ..."
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "System: {{ message['content'] }}\n"
        "{% elif message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "Assistant:"
        "{% endif %}"
    )

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 4. Create Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.01,
        do_sample=True,
        return_full_text=False,
        # Stopping criteria helps prevent the model from rambling
        stop_token_ids=[tokenizer.eos_token_id] 
    )
    
    # 5. Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # CRITICAL FIX: Pass the modified tokenizer explicitly to ChatHuggingFace
    chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)
    
    # 6. Setup Tools
    tool_locals = {"pd": pd, "os": os}
    tools = [PythonAstREPLTool(locals=tool_locals)]
    
    return create_react_agent(chat_model, tools=tools)

# ------------------------------------------------------------------
# 3. ROBUST EVALUATION LOGIC
# ------------------------------------------------------------------

def extract_and_run_code(llm_response: str, local_scope: dict) -> bool:
    """Fallback: Extracts python code blocks and runs them manually."""
    code_blocks = re.findall(r"```python(.*?)```", llm_response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"```(.*?)```", llm_response, re.DOTALL)
    
    if not code_blocks:
        return False

    success = False
    for code in code_blocks:
        try:
            exec(code, local_scope)
            success = True
        except Exception as e:
            # Silent fail on bad blocks, try next
            continue
    return success

def run_single_eval(agent_graph, csv_path: str, run_id: int) -> bool:
    filename = os.path.basename(csv_path)

    # 1. Setup Paths
    abs_input_path = os.path.abspath(csv_path)
    file_dir = os.path.dirname(abs_input_path)
    output_filename = f"analysis_result_run_{run_id}.json"
    abs_output_path = os.path.join(file_dir, output_filename)
    
    if os.path.exists(abs_output_path):
        os.remove(abs_output_path)

    # 2. Calculate Truth
    gold_standard = calculate_gold_standard(abs_input_path)
    if "error" in gold_standard:
        print(f"  [SKIP] Bad CSV: {filename} ({gold_standard['error']})")
        return False

    # 3. Construct Prompts (UPDATED WITH EXACT COLUMNS)
    system_prompt = (
        "You are a Data Engineer.\n"
        "RULES:\n"
        "1. The CSV has columns: C_ID, C_NAME, AGE_YRS, LOC_CD, REG_DT\n"
        "2. Use pandas to read the input CSV.\n"
        "3. Calculate 'total_customers' (count of C_ID) and 'average_age' (mean of AGE_YRS).\n"
        "4. Save them as a JSON object to the exact output file path.\n"
        "5. Enclose your code in ```python``` blocks."
    )
    
    user_prompt = (
        f"Input: '{abs_input_path}'\n"
        f"Output: '{abs_output_path}'\n"
        f"Task: Write and execute the code to generate the JSON now."
    )

    # 4. Invoke Agent
    final_response_text = ""
    try:
        result = agent_graph.invoke({
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        })
        final_response_text = result["messages"][-1].content
    except Exception as e:
        print(f"  File: {filename} -> FAIL [CRASH] ({str(e)[:100]}...)")
        return False

    # 5. VALIDATION & FALLBACK
    
    # Check A: Did the file get created?
    if not os.path.exists(abs_output_path):
        # Fallback: Attempt manual execution of code blocks
        scope = {"pd": pd, "os": os}
        executed = extract_and_run_code(final_response_text, scope)
        
        if not os.path.exists(abs_output_path):
            if executed:
                 print(f"  File: {filename} -> FAIL [EXEC ERROR] (Code ran but file missing)")
            else:
                 print(f"  File: {filename} -> FAIL [NO CODE] (Agent refused to write code)")
            return False

    # Check B: Validate Content
    try:
        with open(abs_output_path, 'r') as f:
            agent_data = json.load(f)

        agent_keys = {k.lower(): v for k, v in agent_data.items()}
        c_exp = gold_standard["total_customers"]
        c_act = agent_keys.get("total_customers")
        a_exp = gold_standard["average_age"]
        a_act = agent_keys.get("average_age", 0)

        if c_act == c_exp and abs(a_act - a_exp) < 0.1:
            print(f"  File: {filename} -> PASS")
            return True
        else:
            print(f"  File: {filename} -> FAIL [WRONG ANSWERS] (Exp: {c_exp}/{a_exp:.2f}, Got: {c_act}/{a_act:.2f})")
            return False

    except Exception as e:
        print(f"  File: {filename} -> FAIL [BAD JSON] ({e})")
        return False

# ------------------------------------------------------------------
# 4. STATISTICAL METRICS
# ------------------------------------------------------------------

def print_enterprise_metrics(run_results: List[Dict]):
    """Calculates pooled accuracy, SD, RSE, and Confidence Intervals."""
    R = len(run_results)
    if R < 2:
        print("Insufficient runs for statistics.")
        return

    Ci_list = [r['correct'] for r in run_results]
    Ti_list = [r['total'] for r in run_results]
    ai_list = [c/t for c, t in zip(Ci_list, Ti_list) if t > 0]

    sum_Ci = sum(Ci_list)
    sum_Ti = sum(Ti_list)
    pooled_accuracy = sum_Ci / sum_Ti if sum_Ti > 0 else 0

    std_dev = np.std(ai_list, ddof=1)
    mean_acc = np.mean(ai_list)
    rse = 1 / math.sqrt(2 * (R - 1))

    df = R - 1
    t_crit = stats.t.ppf(0.975, df)
    margin_err = t_crit * (std_dev / math.sqrt(R))
    
    ci_lower = mean_acc - margin_err
    ci_upper = mean_acc + margin_err

    print("\n" + "="*60)
    print("ENTERPRISE METRICS REPORT")
    print("="*60)
    print(f"Runs (R): {R} | Total Trials: {sum_Ti}")
    print("-" * 60)
    print(f"1. Pooled Accuracy      : {pooled_accuracy:.4f} ({pooled_accuracy*100:.2f}%)")
    print(f"2. Standard Deviation   : {std_dev:.4f}")
    print(f"3. RSE (Uncertainty)    : {rse:.4f} ({rse*100:.1f}%)")
    print(f"4. 95% Conf. Interval   : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("="*60 + "\n")

# ------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ------------------------------------------------------------------

def main():
    try:
        agent = load_agent(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    csv_files = get_csv_files(SANDBOX_DIR)
    if not csv_files:
        print(f"No CSV files found in {SANDBOX_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files. Processing...")
    
    run_stats = []
    
    for i in range(NUM_RUNS):
        print(f"\n--- Starting Run {i+1} of {NUM_RUNS} ---")
        run_correct = 0
        run_total = 0
        
        for csv_file in csv_files:
            run_total += 1
            if run_single_eval(agent, csv_file, run_id=i):
                run_correct += 1
        
        run_stats.append({
            "run_id": i,
            "correct": run_correct,
            "total": run_total
        })

    print_enterprise_metrics(run_stats)

if __name__ == "__main__":
    main()