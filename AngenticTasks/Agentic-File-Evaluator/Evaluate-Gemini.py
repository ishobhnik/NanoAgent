import os
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any
import random
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage 

def create_directory(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return f"OK: created directory {path}"
    except Exception as e:
        return f"ERROR: create_directory {path}: {e}"

def create_file(path: str) -> str:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).touch()
        return f"OK: created file {path}"
    except Exception as e:
        return f"ERROR: create_file {path}: {e}"

def delete_directory(path: str) -> str:
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            return f"OK: deleted directory {path}"
        else:
            return f"OK: directory did not exist {path}"
    except Exception as e:
        return f"ERROR: delete_directory {path}: {e}"

def delete_file(path: str) -> str:
    try:
        if os.path.exists(path):
            os.remove(path)
            return f"OK: deleted file {path}"
        else:
            return f"OK: file did not exist {path}"
    except Exception as e:
        return f"ERROR: delete_file {path}: {e}"

def list_directory(path: str) -> str:
    try:
        if os.path.exists(path):
            contents = os.listdir(path)
            return f"OK: contents of {path}: {contents}"
        else:
            return f"ERROR: list_directory {path}: directory does not exist"
    except Exception as e:
        return f"ERROR: list_directory {path}: {e}"

_TOOL_MAP = {
    "create_directory": create_directory,
    "create_file": create_file,
    "delete_directory": delete_directory,
    "delete_file": delete_file,
    "list_directory": list_directory,
}

class TestTemplateGenerator:
    def __init__(self, artifacts_base: str = "./artifacts"):
        self.artifacts_base = artifacts_base
        self.entity_pool = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta",
            "sigma", "omega", "kappa", "lambda", "phi", "psi", "chi"
        ]
        self.extensions = ["txt", "json", "csv", "xml", "yaml", "conf", "ini"]

    def generate_task_201(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        tasks = []
        for i in range(num_samples):
            qs_id = f"201_{i:03d}"
            entity1 = random.choice(self.entity_pool)
            entity2 = random.choice([e for e in self.entity_pool if e != entity1])
            entity3 = random.choice([e for e in self.entity_pool if e not in [entity1, entity2]])

            template = (
                f"Create the following blank files: {entity1}.log and {entity2}.config "
                f"in the {self.artifacts_base}/{qs_id}/{entity3} directory."
            )

            files_to_check = [
                f"{self.artifacts_base}/{qs_id}/{entity3}/{entity1}.log",
                f"{self.artifacts_base}/{qs_id}/{entity3}/{entity2}.config"
            ]

            tasks.append({
                "question_id": "201",
                "instance_id": qs_id,
                "template": template,
                "files_to_check": files_to_check,
                "scoring_type": "files_exist"
            })
        return tasks

    def generate_task_202(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        tasks = []
        for i in range(num_samples):
            qs_id = f"202_{i:03d}"
            entities = random.sample(self.entity_pool, 9)
            ext1 = random.choice(self.extensions)
            ext2 = random.choice([e for e in self.extensions if e != ext1])
            ext3 = random.choice([e for e in self.extensions if e not in [ext1, ext2]])
            
            expected_structure = [
                f"{self.artifacts_base}/{qs_id}/{entities[0]}/",
                f"{self.artifacts_base}/{qs_id}/{entities[0]}/{entities[1]}/",
                f"{self.artifacts_base}/{qs_id}/{entities[0]}/logs/",
                f"{self.artifacts_base}/{qs_id}/{entities[0]}/logs/{entities[2]}.log",
                f"{self.artifacts_base}/{qs_id}/{entities[3]}/",
                f"{self.artifacts_base}/{qs_id}/{entities[3]}/README.md",
                f"{self.artifacts_base}/{qs_id}/{entities[4]}/",
                f"{self.artifacts_base}/{qs_id}/{entities[4]}/{entities[5]}.{ext1}",
                f"{self.artifacts_base}/{qs_id}/{entities[4]}/{entities[5]}.{ext2}",
                f"{self.artifacts_base}/{qs_id}/{entities[4]}/{entities[5]}.{ext3}"
            ]

            structure_text = "\n".join([f"- {path}" for path in expected_structure])
            template = (
                f"Create this directory structure, including all blank files specified, "
                f"inside the folder '{self.artifacts_base}/{qs_id}':\n{structure_text}"
            )
            files_to_check = [p.rstrip('/') for p in expected_structure]

            tasks.append({
                "question_id": "202",
                "instance_id": qs_id,
                "template": template,
                "expected_structure": files_to_check,
                "scoring_type": "directory_structure"
            })
        return tasks

class AgentEvaluator:
    @staticmethod
    def check_file_exists(filepath: str) -> bool:
        return os.path.isfile(filepath)

    @staticmethod
    def check_directory_exists(dirpath: str) -> bool:
        return os.path.isdir(dirpath)

    def evaluate_task_201(self, task: Dict[str, Any]) -> Dict[str, Any]:
        files_to_check = task["files_to_check"]
        results = {f: self.check_file_exists(f) for f in files_to_check}
        all_correct = all(results.values())
        return {
            "instance_id": task["instance_id"],
            "question_id": task["question_id"],
            "passed": all_correct,
            "details": results,
            "score": 1.0 if all_correct else 0.0
        }

    def evaluate_task_202(self, task: Dict[str, Any]) -> Dict[str, Any]:
        expected_structure = task["expected_structure"]
        results = {}
        for path in expected_structure:
            is_dir = path.endswith('/')
            if is_dir or not Path(path).suffix:
                results[path] = self.check_directory_exists(path.rstrip('/'))
            else:
                results[path] = self.check_file_exists(path)
                
        all_correct = all(results.values())
        return {
            "instance_id": task["instance_id"],
            "question_id": task["question_id"],
            "passed": all_correct,
            "details": results,
            "score": 1.0 if all_correct else 0.0
        }

    def evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if task["scoring_type"] == "files_exist":
            return self.evaluate_task_201(task)
        elif task["scoring_type"] == "directory_structure":
            return self.evaluate_task_202(task)
        else:
            raise ValueError(f"Unknown scoring type: {task['scoring_type']}")
        
class GeminiToolAgent:
    """
    This class asks Gemini to output a strict JSON plan, then executes it.
    Updated to use the modern LangChain .invoke() method with core messages.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
             print("WARNING: GOOGLE_API_KEY not found. Ensure it's set in your environment or .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Try to extract a JSON blob from the model output. Accepts either raw JSON or code fences.
        """
        m_code = re.search(r'```json\s*(\{.*\})\s*```', text, flags=re.DOTALL)
        if m_code:
            return m_code.group(1)
        
        m_raw = re.search(r'(\{.*\})', text, flags=re.DOTALL)

        if m_raw:
            return m_raw.group(1)
        
        return text

    def plan_with_llm(self, instruction: str) -> Dict[str, Any]:
        """
        Ask the LLM to produce a JSON plan using the modern .invoke() method.
        """
        system_prompt = (
            "You are an assistant that converts user instructions into a JSON plan of tool calls.\n"
            "Only respond with JSON (no extra commentary). Follow this exact schema:\n"
            '{\n  "actions": [\n    {"tool": "<tool_name>", "args": ["arg1", ...]},\n    ...\n  ]\n}\n\n'
            "Allowed tools: create_directory, create_file, delete_directory, delete_file, list_directory\n"
            "Tool args are arrays of strings. Paths should be exact. If nothing to do, return {\"actions\": []}."
        )

        user_prompt = f"Instruction: {instruction}\n\nReturn the JSON plan now."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response_message = self.llm.invoke(messages)
            text = response_message.content
            
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

        json_blob = self._extract_json(text)
        try:
            plan = json.loads(json_blob)
            if not isinstance(plan, dict) or "actions" not in plan:
                raise ValueError("Plan JSON missing 'actions' key")
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM JSON plan. Error: {e}\nLLM output:\n{text}")

        return plan

    def execute_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the plan produced by the LLM. Returns a list of results for each action."""
        results = []
        actions = plan.get("actions", [])
        for idx, act in enumerate(actions):
            tool_name = act.get("tool")
            args = act.get("args", [])
            if tool_name not in _TOOL_MAP:
                results.append({"action_index": idx, "tool": tool_name, "args": args, "result": f"ERROR: unknown tool {tool_name}"})
                continue
            try:
                func = _TOOL_MAP[tool_name]
                res = func(*args)
                results.append({"action_index": idx, "tool": tool_name, "args": args, "result": res})
            except Exception as e:
                results.append({"action_index": idx, "tool": tool_name, "args": args, "result": f"EXCEPTION: {e}"})
        return results

    def plan_and_run(self, instruction: str) -> Dict[str, Any]:
        """
        High-level helper: produce plan, run it, and return both the plan and the execution results.
        """
        plan = self.plan_with_llm(instruction)
        exec_results = self.execute_plan(plan)
        return {"plan": plan, "results": exec_results}

def main():
    print("=" * 80)
    print("KAMI-Style File System Agent Evaluation (Gemini planner + local executor)")
    print("=" * 80)

    generator = TestTemplateGenerator(artifacts_base="./artifacts")
    agent = GeminiToolAgent(model_name="gemini-2.5-flash", temperature=0.0)
    evaluator = AgentEvaluator()

    print("\n Generating test tasks...")
    tasks_201 = generator.generate_task_201(num_samples=10)
    tasks_202 = generator.generate_task_202(num_samples=10)
    all_tasks = tasks_201 + tasks_202
    print(f"Generated {len(tasks_201)} instances of Task 201")
    print(f"Generated {len(tasks_202)} instances of Task 202")
    print(f"Total: {len(all_tasks)} tasks")

    print("\n Executing tasks with Gemini-planner...")
    results = []

    for idx, task in enumerate(all_tasks, 1):
        print(f"\n--- Task {idx}/{len(all_tasks)}: {task['instance_id']} ---")
        print(f"Template: {task['template'][:200].replace('\n', ' ')}...")

        try:
            run = agent.plan_and_run(task["template"])
        except Exception as e:
            print(f" Execution failed while planning/executing: {e}")
            results.append({
                **task,
                "execution_success": False,
                "execution_error": str(e),
                "execution_output": None,
                "evaluation": None
            })
            continue

        print("LLM plan:", json.dumps(run["plan"], indent=2)[:1000])
        print("Execution results:")
        for r in run["results"]:
            print(" -", r["result"])

        evaluation = evaluator.evaluate_task(task)
        print(f"{'yes' if evaluation['passed'] else 'no'} Evaluation: {'PASSED' if evaluation['passed'] else 'FAILED'}")
        print(f"Score: {evaluation['score']}")

        results.append({
            **task,
            "execution_success": True,
            "execution_output": run,
            "evaluation": evaluation
        })

    total_tasks = len(results)
    execution_failures = sum(1 for r in results if not r["execution_success"])
    evaluation_passes = sum(1 for r in results if r.get("evaluation") and r["evaluation"]["passed"])
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Tasks: {total_tasks}")
    print(f"Execution Failures: {execution_failures}")
    print(f"Evaluation Passes: {evaluation_passes}")
    print(f"Success Rate: {(evaluation_passes / total_tasks * 100):.2f}%")

    with open("evaluation_results.json", "w") as f:
        summary_results = [{k: v for k, v in r.items() if k != "execution_output"} for r in results]
        json.dump(summary_results, f, indent=2)
    print("\nSaved evaluation_results.json (Note: execution_output omitted for brevity)")

if __name__ == "__main__":
    main()