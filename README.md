# Data Runner
A tool to run large number of data pipeline with asyncio.

Make sure use python 3.10+ to run the code.

## Supported Tasks

- __SequentialTask__: Run tasks in sequence.
- __ParallelTask__: Run tasks in parallel. [under development, not fully tested]
- __IfCondition__: Run different tasks based on a condition. [under development, not fully tested]
- __ForLoopTask__: Run a task multiple times. [under development, not fully tested]
- __WhileLoopTask__: Run a task repeatedly until a condition is met. [under development, not fully tested]
- __MapTask__: Apply a function to each item in a list. [under development, not fully tested]
- __LLMTask__: Call an LLM with a given prompt.
- __WriteFileTask__: Write data to a file.
- __AppendtoListTask__: Append data to a list. [under development, not fully tested]
- __UpdateDictTask__: Update a dictionary. [under development, not fully tested]
- __FnTask__: Apply a function to the input data.

## Usage

```python
from data_runer.tasks import *
from data_runer.llm import LLMCall
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

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

llm_call = LLMCall()
llm_call.register_model("deepseek-r1-250120", api_key=os.getenv("ARK_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3/", n_workers=50)

task = SequentialTask(
    [
        MyLLMTask("question, search_results -> response", params={"model": "deepseek-r1-250120", "temperature": 0.6, "max_tokens": 16384}),
        WriteFileTask("response -> [CONSOLE]"),
    ]
)

pipeline = Pipeline(task)

results = pipeline.batch_call(
    [
        {
            "question": "What is the capital of France?",
            "search_results": [
                "France is a country in Western Europe. Its capital is Paris, which is known for its art, fashion, gastronomy and culture."
            ]
        }
    ]
)

print(results)
```
