import os
import json
import re
import inspect
import shutil
import random
import gc
from pathlib import Path
from typing import List, Dict, Any

# --- Hugging Face and Torch Imports ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- LangChain Core Imports ---
from langchain_core.tools import tool 

# ============================================================================
# TOOLS
# ============================================================================

@tool
def create_directory(path: str) -> str:
    """Create a directory at the specified path."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Successfully created directory: {path}"
    except Exception as e:
        return f"Error creating directory {path}: {str(e)}"

@tool
def create_file(path: str) -> str:
    """Create an empty file at the specified path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).touch()
        return f"Successfully created file: {path}"
    except Exception as e:
        return f"Error creating file {path}: {str(e)}"

@tool
def delete_directory(path: str) -> str:
    """Delete a directory and all its contents."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            return f"Successfully deleted directory: {path}"
        else:
            return f"Directory does not exist: {path}"
    except Exception as e:
        return f"Error deleting directory {path}: {str(e)}"

@tool
def delete_file(path: str) -> str:
    """Delete a file at the specified path."""
    try:
        if os.path.exists(path):
            os.remove(path)
            return f"Successfully deleted file: {path}"
        else:
            return f"File does not exist: {path}"
    except Exception as e:
        return f"Error deleting file {path}: {str(e)}"

@tool
def list_directory(path: str) -> str:
    """List contents of a directory."""
    try:
        if os.path.exists(path):
            contents = os.listdir(path)
            return f"Contents of {path}: {contents}"
        else:
            return f"Directory does not exist: {path}"
    except Exception as e:
        return f"Error listing directory {path}: {str(e)}"

ALL_TOOLS_FUNCTIONS = [create_directory, create_file, delete_directory, delete_file, list_directory]

def tool_to_qwen_schema(tool_obj) -> Dict[str, Any]:
    name = tool_obj.name
    description = tool_obj.description
    
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
    
    sig = inspect.signature(tool_obj.func)
    for param_name, _ in sig.parameters.items():
        schema["function"]["parameters"]["properties"][param_name] = {
            "type": "string",
            "description": f"The argument for {name}"
        }
        schema["function"]["parameters"]["required"].append(param_name)
        
    return schema

QWEN_TOOL_SCHEMAS = [tool_to_qwen_schema(t) for t in ALL_TOOLS_FUNCTIONS]
TOOL_FUNCTION_MAP = {t.name: t.func for t in ALL_TOOLS_FUNCTIONS}


# ============================================================================
# TEMPLATES
# ============================================================================

class TestTemplateGenerator:
    def __init__(self, artifacts_base: str = "./artifacts"):
        self.artifacts_base = artifacts_base
        self.entity_pool = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    
    def generate_task_201(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        tasks = []
        for i in range(num_samples):
            qs_id = f"201_{i:09d}"
            e1, e2, e3 = random.sample(self.entity_pool, 3)
            print(qs_id)
            template = f"Create the following blank files: {e1}.log and {e2}.config in the {self.artifacts_base}/{qs_id}/{e3} directory."
            tasks.append({
                "question_id": "201", "instance_id": qs_id, "template": template,
                "files_to_check": [f"{self.artifacts_base}/{qs_id}/{e3}/{e1}.log", f"{self.artifacts_base}/{qs_id}/{e3}/{e2}.config"],
                "scoring_type": "files_exist"
            })
        return tasks
    
    def generate_task_202(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        tasks = []
        for i in range(num_samples):
            qs_id = f"202_{i:09d}"
            e = random.sample(self.entity_pool, 6)
            struct = [
                f"{self.artifacts_base}/{qs_id}/{e[0]}/",
                f"{self.artifacts_base}/{qs_id}/{e[0]}/logs/{e[1]}.log",
                f"{self.artifacts_base}/{qs_id}/{e[2]}/README.md"
            ]
            template = f"Create directory structure inside '{self.artifacts_base}/{qs_id}': folder '{e[0]}' containing 'logs/{e[1]}.log', and folder '{e[2]}' containing 'README.md'."
            tasks.append({
                "question_id": "202", "instance_id": qs_id, "template": template,
                "expected_structure": struct, "scoring_type": "directory_structure"
            })
        return tasks


class AgentEvaluator:
    @staticmethod
    def check_file_exists(filepath: str) -> bool:
        return os.path.isfile(filepath)
    
    @staticmethod
    def check_directory_exists(dirpath: str) -> bool:
        return os.path.isdir(dirpath)
    
    def evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if task["scoring_type"] == "files_exist":
            details = {path: self.check_file_exists(path) for path in task["files_to_check"]}
        elif task["scoring_type"] == "directory_structure":
            details = {}
            for path in task["expected_structure"]:
                if path.endswith('/'): details[path] = self.check_directory_exists(path.rstrip('/'))
                else: details[path] = self.check_file_exists(path)
        
        passed = all(details.values())
        return {"passed": passed, "details": details, "score": 1.0 if passed else 0.0}


# ============================================================================
# AGENT
# ============================================================================

class FileSystemAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cpu"
        print(f"\n🔧 Initializing {model_name} on {self.device.upper()}...")
        
        # CHANGED: Use float16 to save memory (Fixes OOM on 6GB cards)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.tool_schemas = QWEN_TOOL_SCHEMAS
        self.tool_map = TOOL_FUNCTION_MAP
    
    def execute_task(self, task_template: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. You have access to tools. Use the tools by outputting a JSON block with 'name' and 'arguments'. If no tool is needed, output the final answer."},
            {"role": "user", "content": task_template}
        ]
        
        final_output = None
        
        for i in range(10): 
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, tools=self.tool_schemas
                )
                
                inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=512, do_sample=False, temperature=0.0
                    )
                
                output = self.tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                
                # --- PARSING FIXES ---
                # 1. Clean up XML tags often emitted by Qwen (e.g. <tool_call>)
                clean_output = re.sub(r'<[^>]+>', '', output).strip()
                
                # 2. Find JSON anywhere in the string
                # This finds the first { ... } block
                match = re.search(r'(\{.*\})', clean_output, re.DOTALL)
                
                if not match:
                    # Fallback: Look for json inside markdown blocks
                    match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)

                tool_found = False
                if match:
                    try:
                        json_str = match.group(1)
                        # Attempt to sanitize common bad JSON (e.g., trailing commas)
                        json_str = json_str.replace("'", '"') 
                        
                        tool_call_data = json.loads(json_str)
                        if isinstance(tool_call_data, list): tool_call_data = tool_call_data[0]
                        
                        tool_name = tool_call_data.get('name')
                        tool_args = tool_call_data.get('arguments', {})
                        
                        if tool_name in self.tool_map:
                            tool_found = True
                            print(f"\n> Tool Call: {tool_name}({tool_args})")
                            tool_output = self.tool_map[tool_name](**tool_args)
                            print(f"< Tool Output: {tool_output}")
                            
                            messages.append({"role": "assistant", "content": output})
                            messages.append({"role": "tool", "content": tool_output, "name": tool_name})
                    except Exception as e:
                        # Parsing failed, but maybe it wasn't a tool call
                        pass 

                if not tool_found:
                    final_output = output
                    break
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": True, "output": final_output}

# ============================================================================
# MAIN
# ============================================================================

def main():
    generator = TestTemplateGenerator()
    
    try:
        agent = FileSystemAgent(model_name="Qwen/Qwen2.5-1.5B-Instruct") 
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    evaluator = AgentEvaluator()
    
    # 20 samples
    tasks = generator.generate_task_201(10) + generator.generate_task_202(10)
    
    print(f"Running {len(tasks)} tasks...")
    
    for idx, task in enumerate(tasks):
        # Memory Cleanup: Critical for 6GB GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        shutil.rmtree(f"./artifacts/{task['instance_id']}", ignore_errors=True)
        
        print(f"\n--- Task {idx+1}: {task['instance_id']} ---")
        # print(f"Goal: {task['template']}")
        
        res = agent.execute_task(task["template"])
        
        if res["success"]:
            eval_res = evaluator.evaluate_task(task)
            print(f"Result: {'✅ PASSED' if eval_res['passed'] else '❌ FAILED'}")
        else:
            print(f"Error: {res['error']}")

if __name__ == "__main__":
    main()