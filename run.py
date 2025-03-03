
import json
import os
import re
import time
import shutil
from typing import Any, Dict, List, Optional, Tuple
from data_runer.tasks import FnTask, LLMTask, SequentialTask, WriteFileTask
from data_runer.pipeline import Pipeline
from data_runer.utils import get_to_file_writer_error_handler, ThreadSafeFileWriter
from data_runer.llm import LLMCall

from pypinyin import lazy_pinyin, Style
from pypinyin_dict.pinyin_data import ktghz2013
from pypinyin_dict.phrase_pinyin_data import large_pinyin

# load better pinyin data
ktghz2013.load()
# load better phrase pinyin data
large_pinyin.load()

# The error description of GEC part is from FCGEC dataset
task_prompt = """
# The following contents are the search results related to the user's message:
{search_results}
In the search results I provide to you, each result is formatted as [webpage X begin]...[webpage X end], where X represents the numerical index of each article. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
When responding, please keep the following points in mind:
- Today is {cur_date}.
- Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.
- For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the search results unless necessary.
- For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the search results, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
- If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.
- For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.
- Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.
- Your answer should synthesize information from multiple relevant webpages and avoid repeatedly citing the same webpage.
- Unless the user requests otherwise, your response should be in the same language as the user's question.

# The user's message is:
{question}
""".strip()

class MyLLMTask(LLMTask):
    def preprocess(self, **data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        question = data["question"]
        search_results = data["search_results"]
        str_search_results = ""
        for i, search_result in enumerate(search_results):
            str_search_results += f"[webpage {i} begin]{search_result}[webpage {i} end]\n"
        cur_date = datetime.now().strftime("%Y-%m-%d")
        messages = [
            {"role": "user", "content": task_prompt.format(question=question, search_results=str_search_results, cur_date=cur_date)}
        ]
        return messages, None

def parse_response(response: str) -> Dict[str, Any]:
    return response["content"], response["reasoning_content"], response["stop_reason"]

def get_output(
        id: str,
        question: str, 
        content: str,
        reasoning_content: str, 
        stop_reason: str) -> Dict[str, Any]:
    return json.dumps({
        "id": id,
        "question": question,
        "content": content,
        "reasoning": reasoning_content,
        "stop_reason": stop_reason,
    }, ensure_ascii=False) + "\n"

def get_finished_str() -> str:
    return "Finished"

def check_exact_match(prediction: str, references: List[str]) -> bool:
    return prediction in references

if __name__ == "__main__":
    llm_call = LLMCall()
    llm_call.register_model("deepseek-r1-250120", api_key=os.getenv("ARK_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3/", n_workers=50)

    input_file = "datasets/data.jsonl"
    output_file = "results/output.jsonl"
    resume_unfinished = True
    
    finished_ids = set()
    if os.path.exists(output_file) and resume_unfinished:
        # copy the output file to a tmp file
        tmp_file = output_file.replace(".jsonl", f".{time.time()}.jsonl")
        shutil.copyfile(output_file, tmp_file)
        # read and record the finished ids
        with ThreadSafeFileWriter(output_file) as file_writer:
            with open(tmp_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        finished_ids.add(data["id"])
                        file_writer.write(line)
    print(f"Finished ids: {len(finished_ids)}")

    task = SequentialTask(
        [
            MyLLMTask("question, search_results -> response", params={"model": "deepseek-r1-250120", "temperature": 0.6, "max_tokens": 16384}),
            WriteFileTask("response -> [CONSOLE]"),
            FnTask("response -> content, reasoning_content, stop_reason", parse_response),
            FnTask("id, question, content, reasoning_content, stop_reason -> output", get_output),
            WriteFileTask("output -> [FILE]", output_file),
        ],
        n_retries=3,
        error_handling=get_to_file_writer_error_handler(output_file.replace(".jsonl", ".error.log"))
    )
    pipeline = Pipeline(task)

    with open(input_file, "r") as f:
        batch_data = [json.loads(line) for line in f]
        print(f"Total data: {len(batch_data)}")
        batch_data = [data for data in batch_data if data["id"] not in finished_ids]
        print(f"Unfinished data: {len(batch_data)}")

    start_time = time.time()
    results = pipeline.batch_call(batch_data)
    end_time = time.time()
    print(f"Time (with async): {end_time - start_time}")
    print(f"Average time (with async): {(end_time - start_time) / len(batch_data)}")
    with open(output_file.replace(".jsonl", ".finished.jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
