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
```
