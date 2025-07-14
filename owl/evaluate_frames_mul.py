import argparse
import json
import os
import time
from typing import List, Dict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from examples.run_deepseek_zh import construct_society,run_society
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import markdown     # pip install markdown
from bs4 import BeautifulSoup  # pip install beautifulsoup4
# TIMEOUT_SECONDS = 60
TIMEOUT_SECONDS = 360
import re

# from deepseek import DeepSeekAPI

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="http://localhost:8000/v1")
# client = OpenAI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
# client = DeepSeekAPI(api_key=os.getenv("DEEPSEEK_API_KEY"))
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
    # base_url="https://www.chataiapi.com/v1"
)

SLEEP_INTERVAL = 300


def load_existing_results(filename: str) -> List[Dict]:
    try:
        # 指定 utf-8 编码读取
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_result(filename: str, result: Dict):
    results = load_existing_results(filename)
    results.append(result)
    # with open(filename, 'w') as f:
    #     json.dump(results, f, indent=2)
    results.sort(key=lambda x: x["index"])
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
def save_results(filename: str, results: List[Dict]):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def get_last_processed_index(results: List[Dict]) -> int:
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)


def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> str:
    return f"Here are the relevant Wikipedia articles:\n{wiki_links}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n"


# def get_llm_response(prompt: str, model: str) -> str:
#     response = client.with_options(timeout=1000.0).chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=1000,
#         n=1,
#         stop=None,
#         temperature=0.7,
#         extra_body={"optillm_approach": "readurls&memory"}
#     )
#     return response.choices[0].message.content.strip()


def get_llm_response(prompt: str, model: str) -> str:
    """
    ignore `model` arg here, we call camel_answer instead.
    """
    try:
        society = construct_society(prompt)
        answer, _, _ = run_society(society)
        # return camel_answer(prompt, output_lang="English")
        return answer
    except Exception as e:
        # 若 CAMEL 抛错，返回空串，继续下一条
        print(f"[CAMEL ERROR] {e}")
        return ""


def markdown_to_text(md: str) -> str:
    """
    1) 先用 python-markdown 渲染成 HTML
    2) 再用 BeautifulSoup 提取纯文本
    3) 最后用正则做一些多余空行、空格的清理
    """
    # 渲染成 HTML
    html = markdown.markdown(md)
    # 提取纯文本
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    # 合并多余空行
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text

def evaluate_response(question: str, llm_response: str, ground_truth: str, model: str) -> Dict[str, str]:
    # llm_response = markdown_to_text(llm_response)

    evaluation_prompt = f"""===Task===
I need your help in evaluating an answer provided by an LLM against a ground
truth answer. Your task is to determine if the ground truth answer is present in the LLM's
response. Please analyze the provided data and make a decision.Note that the Predicted Answer in Input Data is in markdown format, so avoid its formatting when assessing it.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers.
Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the
"Ground Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {llm_response}
- Ground Truth Answer: {ground_truth}
===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )
Please proceed with the evaluation."""
#     evaluation_prompt = f"""===Task===
# I need your help in evaluating an answer provided by an LLM against a ground
# truth answer. Your task is to determine if the ground truth answer is present in the LLM's
# response. Please analyze the provided data and make a decision.
#
# Important:
# The Predicted Answer is written in GitHub-flavored Markdown.
# Before comparing, mentally strip away all Markdown syntax — e.g. headings
# (`#`, `##`), lists (`-`, `*`, `1.`), links (`[text](url)`), images,
# inline/​block code (`` `code` `` / ``` code ```), LaTeX ($...$, $$...$$),
# and emphasis markers (`*`, `_`, `**`).
# Focus solely on the underlying semantic content.
#
# If a code block, table, or math expression itself contains relevant facts
# (e.g. numeric results in a table), you should read the text/numbers inside
# and include them in your judgment.
#
# ===Instructions===
# 1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
# 2. Consider the substance of the answers - look for equivalent information or correct answers.
# Do not focus on exact wording unless the exact wording is crucial to the meaning.
# 3. Your final decision should be based on whether the meaning and the vital facts of the
# "Ground Truth Answer" are present in the "Predicted Answer:"
# ===Input Data===
# - Question: {question}
# - Predicted Answer: {llm_response}
# - Ground Truth Answer: {ground_truth}
# ===Output Format===
# Provide your final evaluation in the following format:
# "Explanation:" (How you made the decision?)
# "Decision:" ("TRUE" or "FALSE" )
# Please proceed with the evaluation."""

    evaluation_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": evaluation_prompt}
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.3,
    )

    evaluation_text = evaluation_response.choices[0].message.content.strip()

    # Extract the decision and explanation
    lines = evaluation_text.split('\n')
    decision = "FALSE"
    explanation = ""
    for line in lines:
        if line.startswith("Decision:"):
            decision = line.split(":")[1].strip().upper()
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()

    return {"decision": decision, "explanation": explanation}


# def process_item(args):
# def process_item(item, model, last_idx):
def process_item(item, model):
    """
    多进程 worker：处理单条样本，返回结果 dict 或 None（已处理过的）。
    args = (item, model, last_idx)
    """
    # item, model, last_idx = args
    idx = int(item["Unnamed: 0"])
    # if idx <= last_idx:
    #     return None
    prompt = generate_llm_prompt(item["Prompt"], item["wiki_links"])
    resp = get_llm_response(prompt, model)
    ev   = evaluate_response(item["Prompt"], resp, item["Answer"], model)
    return {
        "index": idx,
        "prompt": item["Prompt"],
        "ground_truth": item["Answer"],
        "llm_response": resp,
        "evaluation_decision": ev["decision"],
        "evaluation_explanation": ev["explanation"],
        "reasoning_type": item["reasoning_types"],
    }

def main(model: str):
    # 1. 载入数据集
    dataset = list(load_dataset("google/frames-benchmark", split="test"))

    # 2. 断点续跑：读取已存 json，并计算 last_idx
    filename = f"evaluation_results_{model.replace('/', '_')}.json"
    existing = load_existing_results(filename)
    last_idx = get_last_processed_index(existing)
    processed_indices = {r["index"] for r in existing}
    # 3. 构造仅包含未处理样本的任务列表
    # tasks = [(item, model, last_idx) for item in dataset if int(item["Unnamed: 0"]) > last_idx]
    tasks = [item for item in dataset if int(item["Unnamed: 0"]) not in processed_indices]


    from signal import signal, SIGINT, SIG_IGN
    orig_handler = signal(SIGINT, SIG_IGN)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        signal(SIGINT, orig_handler)
        futures = {
            # executor.submit(process_item, item, model, last_idx): item
            # for item, model, last_idx in tasks
            executor.submit(process_item, item, model): item
            for item in tasks
        }

        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    res = fut.result(timeout=TIMEOUT_SECONDS)
                except TimeoutError:
                    # 单个子任务超时则跳过
                    continue
                if res:
                    save_result(filename, res)
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，已中断，当前进度已保存。")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on google/frames-benchmark")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini)")
    args = parser.parse_args()

    main(args.model)