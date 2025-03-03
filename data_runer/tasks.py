
"""
The code design is inspired and largely copied from:
https://github.com/stanfordnlp/dspy/blob/main/dspy/signatures/signature.py#L333
"""

import json
import random
import re
import sys
import asyncio
import time
import traceback
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from data_runer.utils import ThreadSafeFileWriter
from data_runer.llm import LLMCall
from data_runer.executors import ThreadPoolExecutor

max_workers = 100
max_tasks = 100
the_executor = ThreadPoolExecutor(max_workers)
the_semaphore = [
    asyncio.Semaphore(max_workers) for _ in range(max_tasks)
]


class TaskBase:
    """
    Base class for all tasks in the data runner framework.
    
    This class provides the foundation for defining tasks with input and output fields,
    validation, and execution mechanisms.
    
    Attributes:
        signature (str): A string representation of the task's input and output fields.
        input_fields (List[str]): List of input field names required by the task.
        output_fields (List[str]): List of output field names produced by the task.
        n_retries (int): The number of retries for the task.
        error_handling (Callable[[Exception, Dict[str, Any]], Any]): A function to handle errors. It should take two arguments: the error message and the current context.
    """
    
    def __init__(self, signature: str, n_retries: int = 0, error_handling: Optional[Callable[[Exception], Any]] = None):
        """
        Initialize a task with a signature defining its inputs and outputs.
        
        Args:
            signature (str): A string in the format "input1,input2,...->output1,output2,..."
                             that defines the task's input and output fields.
        """
        self.signature = signature
        self.n_retries = n_retries
        self.error_handling = error_handling
        self.__parse_signature()

    def __parse_signature(self) -> Tuple[str, str]:
        """
        Parse the signature string into input and output fields.
        
        Raises:
            ValueError: If the signature format is invalid.
            
        Returns:
            Tuple[str, str]: A tuple containing the input and output parts of the signature.
        """
        if self.signature.count("->") != 1:
            raise ValueError(f"Invalid signature format: '{self.signature}', must contain exactly one '->'.")

        inputs_str, outputs_str = self.signature.split("->")

        self.input_fields = [field.strip() for field in inputs_str.split(",") if field.strip()]
        self.output_fields = [field.strip() for field in outputs_str.split(",") if field.strip()]

    def __get_inputs_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the visible fields from the context for the given task.
        """
        return {k: v for k, v in context.items() if k in self.input_fields}
    
    def _update_context(self, context: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the context with the result of the given task.
        """
        if len(self.output_fields) == 1:
            context[self.output_fields[0]] = result
        elif len(self.output_fields) > 1:
            assert len(self.output_fields) == len(result)
            context.update({k: v for k, v in zip(self.output_fields, result)})
        if '@task_step' not in context:
            context['@task_step'] = 1
        else:
            context['@task_step'] += 1
        # if there is no output field, we do nothing
        return context

    def run(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the task with the given input data.
        
        This method must be implemented by subclasses.
        
        Args:
            data (Dict[str, Any]): The input data for the task.
            
        Returns:
            Dict[str, Any]: The output data produced by the task.
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Running is not implemented for this task.")

    def __call__(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the task as a function, validating inputs and outputs.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: The output data produced by the task.
        """
        data = self.__get_inputs_from_context(context)
        for i in range(self.n_retries + 1):
            try:
                result = self.run(**data)
                context = self._update_context(context, result)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    else:
                        raise e
                else:
                    continue

    async def async_call(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously call the task, running it in a separate thread.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: The output data produced by the task.
        """
        loop = asyncio.get_event_loop()
        # with concurrent.futures.ThreadPoolExecutor(max_workers=100) as pool:
        task_step = context.get('@task_step', 0)
        priority = -task_step
        random_num = random.random()
        priority_tuple = (priority, time.time(), random_num)
        async with the_semaphore[task_step % len(the_semaphore)]:
            print(f"Added task {self.__class__.__name__} with priority {priority_tuple}")
            result = await loop.run_in_executor(the_executor, partial(self, **context), priority_tuple)
            print(f"Task {self.__class__.__name__} with priority {priority_tuple} finished")
        return result

class ControlFlowTask(TaskBase):
    """
    Base class for tasks that control the flow of execution.
    
    This class provides utility methods for handling control flow operations
    like conditionals and loops.
    """

    def _merge_context(self, context: Dict[str, Any], new_context: Dict[str, Any], merge_fields: List[str]) -> Dict[str, Any]:
        """
        Merge the new context with the existing context.
        """
        for field in merge_fields:
            context[field] = new_context[field]
        if '@task_step' not in context:
            context['@task_step'] = new_context.get('@task_step', 1)
        else:
            context['@task_step'] = max(new_context.get('@task_step', 1), context['@task_step'])
        return context

class SequentialTask(ControlFlowTask):
    """
    A task that executes a sequence of tasks in order.
    
    Each task in the sequence receives the output of the previous task as input.
    
    Attributes:
        tasks (List[TaskBase]): The list of tasks to execute sequentially.
    """
    
    def __init__(self, tasks: List[TaskBase], **kwargs: Dict[str, Any]):
        """
        Initialize a sequential task with a list of subtasks.
        
        Args:
            tasks (List[TaskBase]): The list of tasks to execute sequentially.
            n_retries (int): The number of retries for each task.
        """
        self.tasks = tasks
        all_input_fields = set(sum([task.input_fields for task in tasks], []))
        all_output_fields = set(sum([task.output_fields for task in tasks], []))
        super().__init__(f"{','.join(all_input_fields)}->{','.join(all_output_fields)}", **kwargs)
        if self.error_handling is not None:
            # recursively add error handling to all tasks
            for task in self.tasks:
                task.error_handling = self.error_handling

    def __call__(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all tasks in sequence, passing the output of each task as input to the next.
        
        Args:
            context (Dict[str, Any]): The initial context.
            
        Returns:
            Dict[str, Any]: The final output data after all tasks have executed.
        """
        for i in range(self.n_retries + 1):
            try:
                for task in self.tasks:
                    context = task(**context)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    else:
                        raise e
                else:
                    continue


    async def async_call(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously execute all tasks in sequence.
        
        Args:
            context (Dict[str, Any]): The initial context.
            
        Returns:
            Dict[str, Any]: The final output data after all tasks have executed.
        """
        for i in range(self.n_retries + 1):
            try:
                for task in self.tasks:
                    context = await task.async_call(**context)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    else:
                        raise e
                else:
                    continue

class IfCondition(ControlFlowTask):
    """
    A task that conditionally executes one of two branches based on a condition.
    
    Attributes:
        condition (TaskBase): The task that determines which branch to execute.
        true_branch (TaskBase): The task to execute if the condition is true.
        false_branch (TaskBase): The task to execute if the condition is false.
    """
    
    def __init__(self, condition: TaskBase, true_branch: TaskBase, false_branch: TaskBase, **kwargs: Dict[str, Any]):
        """
        Initialize an if-condition task.
        
        Args:
            condition (TaskBase): The task that determines which branch to execute.
            true_branch (TaskBase): The task to execute if the condition is true.
            false_branch (TaskBase): The task to execute if the condition is false.
            n_retries (int): The number of retries for each task.
        """
        self.condition = condition
        assert len(condition.output_fields) == 1, "Condition task must have exactly one output field."
        self.true_branch = true_branch
        self.false_branch = false_branch
        all_input_fields = set(condition.input_fields) | set(true_branch.input_fields) | set(false_branch.input_fields)
        all_output_fields = set(true_branch.output_fields) | set(false_branch.output_fields)
        super().__init__(f"{','.join(all_input_fields)}->{','.join(all_output_fields)}", **kwargs)
        if self.error_handling is not None:
            # recursively add error handling to all tasks
            self.true_branch.error_handling = self.error_handling
            self.false_branch.error_handling = self.error_handling

    def __call__(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute either the true or false branch based on the condition.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: The output data from the executed branch.
        """
        for i in range(self.n_retries + 1):
            try:
                context = self.condition(context)
                if context[self.condition.output_fields[0]]:
                    context = self.true_branch(context)
                    return context
                else:
                    context = self.false_branch(context)
                    return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    else:
                        raise e
                else:
                    continue

    async def async_call(self, **data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously execute either the true or false branch based on the condition.
        
        Args:
            data (Dict[str, Any]): The input data.
            
        Returns:
            Dict[str, Any]: The output data from the executed branch.
        """
        for i in range(self.n_retries + 1):
            try:
                context = await self.condition.async_call(data)
                if context[self.condition.output_fields[0]]:
                    context = await self.true_branch.async_call(data)
                else:
                    context = await self.false_branch.async_call(data)
                    return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), data)
                    else:
                        raise e
                else:
                    continue

class ForLoopTask(ControlFlowTask):
    """
    A task that executes a body task for each item in an iterable input field.
    
    Attributes:
        input_fields (str): The name of the field containing the iterable.
        loop_var (str): The name of the variable to assign each item to.
        body (TaskBase): The task to execute for each item.
    """
    
    def __init__(self, input_fields: str, loop_var: str, body: TaskBase, **kwargs: Dict[str, Any]):
        """
        Initialize a for-loop task.
        
        Args:
            input_fields (str): The name of the field containing the iterable.
            loop_var (str): The name of the variable to assign each item to.
            body (TaskBase): The task to execute for each item.
            n_retries (int): The number of retries for each task.
        """
        self.input_fields = input_fields
        self.loop_var = loop_var
        self.body = body
        all_input_fields = {set(body.input_fields) | {input_fields,} - {loop_var,}}
        all_output_fields = set(body.output_fields)
        super().__init__(f"{','.join(all_input_fields)}->{','.join(all_output_fields)}", **kwargs)
        if self.error_handling is not None:
            # recursively add error handling to all tasks
            self.body.error_handling = self.error_handling

    def __call__(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the body task for each item in the iterable input field.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: The final output data after all iterations.
        """
        for i in range(self.n_retries + 1):
            try:
                for loop_var in context[self.input_fields]:
                    context[self.loop_var] = loop_var
                    context = self.body(context)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue
    
    async def async_call(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously execute the body task for each item in the iterable input field.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: The final output data after all iterations.
        """
        for i in range(self.n_retries + 1):
            try:
                for loop_var in context[self.input_fields]:
                    context[self.loop_var] = loop_var
                    context = await self.body.async_call(context)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue

class WhileLoopTask(ControlFlowTask):
    """
    A task that repeatedly executes a body task while a condition is true.
    
    Attributes:
        condition (TaskBase): The task that determines whether to continue looping.
        body (TaskBase): The task to execute in each iteration.
    """
    
    def __init__(self, condition: TaskBase, body: TaskBase, **kwargs: Dict[str, Any]):
        """
        Initialize a while-loop task.
        
        Args:
            condition (TaskBase): The task that determines whether to continue looping.
            body (TaskBase): The task to execute in each iteration.
            n_retries (int): The number of retries for each task.
        """
        self.condition = condition
        assert len(condition.output_fields) == 1, "Condition task must have exactly one output field."
        self.body = body
        all_input_fields = set(condition.input_fields) | set(body.input_fields)
        all_output_fields = set(body.output_fields)
        super().__init__(f"{','.join(all_input_fields)}->{','.join(all_output_fields)}", **kwargs)
        if self.error_handling is not None:
            # recursively add error handling to all tasks
            self.body.error_handling = self.error_handling
            self.condition.error_handling = self.error_handling

    def __call__(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the body task repeatedly while the condition is true.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: The final output data after all iterations.
        """
        for i in range(self.n_retries + 1):
            try:
                context = self.condition(context)
                while context[self.condition.output_fields[0]]:
                    context = self.body(context)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue
    
    async def async_call(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously execute the body task repeatedly while the condition is true.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: The final output data after all iterations.
        """
        for i in range(self.n_retries + 1):
            try:
                context = await self.condition.async_call(context)
                while context[self.condition.output_fields[0]]:
                    context = await self.body.async_call(context)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue

class MapTask(ControlFlowTask):
    """
    A task that applies a mapping function to each element in an input field.
    
    Attributes:
        map_fn (TaskBase): The task to apply to each element.
    """
    
    def __init__(self, signature: str, map_fn: TaskBase, **kwargs: Dict[str, Any]):
        """
        Initialize a map task.
        
        Args:
            signature (str): The signature defining input and output fields.
            map_fn (TaskBase): The task to apply to each element.
            n_retries (int): The number of retries for each task.
        """
        self.map_fn = map_fn
        super().__init__(signature, **kwargs)
        assert len(self.map_fn.output_fields) == 1, "Map function must have exactly one output field."
        if self.error_handling is not None:
            # recursively add error handling to all tasks
            self.map_fn.error_handling = self.error_handling

    def __wrap_item(self, context: Dict[str, Any], item: Any) -> Dict[str, Any]:
        """
        Wrap an item from the input iterable as input data for the map function.
        
        Args:
            item (Any): An item from the input iterable.
            
        Returns:
            Dict[str, Any]: The wrapped item as input data.
        """
        context = context.copy()
        for k, v in zip(self.map_fn.input_fields, item):
            context[k] = v
        return context
        
    def __get_map_result(self, context: Dict[str, Any]) -> Any:
        """
        Get the result from the map function.
        """
        if len(self.map_fn.output_fields) > 1:
            return tuple(context[k] for k in self.map_fn.output_fields)
        else:
            return context[self.map_fn.output_fields[0]]

    def __call__(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the map function to each element in the input field.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: A list of results from applying the map function.
        """
        max_step = context.get('@task_step', 0)
        for i in range(self.n_retries + 1):
            try:
                results = []
                for item in zip(*[context[field] for field in self.input_fields]):
                    this_context = self.__wrap_item(context, item)
                    this_context = self.map_fn(this_context)
                    results.append(self.__get_map_result(this_context))
                    max_step = max(max_step, this_context.get('@task_step', 0))
                context[self.output_fields[0]] = results
                context['@task_step'] = max_step
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue
    
    async def async_call(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously apply the map function to each element in the input field.
        
        Args:
            context (Dict[str, Any]): The context.
            
        Returns:
            Dict[str, Any]: A list of results from applying the map function.
        """
        for i in range(self.n_retries + 1):
            try:
                results = await asyncio.gather(*[self.map_fn.async_call(**self.__wrap_item(context, item)) for item in zip(*[context[field] for field in self.input_fields])])
                context[self.output_fields[0]] = results
                context['@task_step'] = max(context.get('@task_step', 1), max(this_context.get('@task_step', 1) for this_context in results))
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue

class ParallelTask(ControlFlowTask):
    """
    A task that executes multiple tasks in parallel.
    """
    def __init__(self, tasks: List[TaskBase], **kwargs: Dict[str, Any]):
        self.tasks = tasks
        # ensure there is no overlap in output fields and input fields
        all_input_fields = sum([task.input_fields for task in tasks], [])
        all_output_fields = sum([task.output_fields for task in tasks], [])
        # check if there is multiple tasks with the same output field
        assert len(set(all_output_fields)) == len(all_output_fields), "There is multiple tasks with the same output field. It will cause write conflict."
        assert len(set(all_input_fields) & set(all_output_fields)) == 0, "There is overlap in input fields and output fields. It can not be parallelized."
        super().__init__(f"{','.join(all_input_fields)}->{','.join(all_output_fields)}", **kwargs)
        if self.error_handling is not None:
            # recursively add error handling to all tasks
            for task in self.tasks:
                task.error_handling = self.error_handling

    def __call__(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tasks in parallel. But when it not in async mode, it will execute the tasks sequentially.
        """
        for i in range(self.n_retries + 1):
            try:
                for task in self.tasks:
                    result = task(**context)
                    context = self._merge_context(context, result, task.output_fields)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue

    async def async_call(self, **context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously execute the tasks in parallel.
        """
        for i in range(self.n_retries + 1):
            try:
                results = await asyncio.gather(*[task.async_call(**context) for task in self.tasks])
                for result, task in zip(results, self.tasks):
                    context = self._merge_context(context, result, task.output_fields)
                return context
            except Exception as e:
                if i == self.n_retries:
                    if self.error_handling is not None:
                        self.error_handling(traceback.format_exc(), context)
                    raise e
                else:
                    continue

class FnTask(TaskBase):
    """
    A task that wraps a Python function.
    
    Attributes:
        fn (Callable): The function to execute.
    """
    
    def __init__(self, signature: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]], **kwargs: Dict[str, Any]):
        """
        Initialize a function task.
        
        Args:
            signature (str): The signature defining input and output fields.
            fn (Callable): The function to execute.
        """
        super().__init__(signature, **kwargs)
        self.fn = fn

    def run(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the wrapped function with the given input data.
        
        Args:
            data (Dict[str, Any]): The input data.
            
        Returns:
            Dict[str, Any]: The output data from the function.
        """
        return self.fn(**kwargs)

class WriteFileTask(TaskBase):
    """
    A task that writes data to a file, console, or error stream.
    
    This task is thread-safe and manages file handles to ensure proper resource usage.
    
    Attributes:
        file_path (str): The path to the file to write to.
    """

    def __init__(self, signature: str, file_path: str = None, **kwargs: Dict[str, Any]):
        """
        Initialize a file writing task.
        
        Args:
            signature (str): The signature defining input and output fields.
            file_path (str): The path to the file to write to.
        """
        super().__init__(signature, **kwargs)
        assert len(self.input_fields) == 1, "WriteFileTask only supports one input field."
        assert len(self.output_fields) == 1
        assert self.output_fields[0] in {"[FILE]", "[CONSOLE]", "[ERROR]"}
        self.file_path = file_path
        if self.output_fields[0] == "[FILE]":
            if file_path is None:
                raise ValueError("file_path is required when output_fields[0] is '[FILE]'.")
            self.file_writer = ThreadSafeFileWriter(file_path)

    def run(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write the input data to the specified output destination.
        
        Args:
            kwargs (Dict[str, Any]): The data to write.
            
        Returns:
            Dict[str, Any]: The output data (typically empty).
        """
        if self.output_fields[0] == "[FILE]":
            self.file_writer.write(str(kwargs[self.input_fields[0]]))
            self.file_writer.flush()
        elif self.output_fields[0] == "[CONSOLE]":
            print(kwargs[self.input_fields[0]])
        elif self.output_fields[0] == "[ERROR]":
            print(kwargs[self.input_fields[0]], file=sys.stderr)

class AppendtoListTask(TaskBase):
    """
    A task that appends data to a list.
    """
    def __init__(self, signature: str, **kwargs: Dict[str, Any]):
        super().__init__(signature, **kwargs)
        assert len(self.output_fields) == 1
        self.list_field = self.output_fields[0]
        self.true_input_fields = self.input_fields
        # add the list field to the input fields, to ensure the visibility of the list field
        self.input_fields = self.true_input_fields + [self.list_field]

    def run(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs[self.list_field].append(tuple(kwargs[k] for k in self.true_input_fields))
        return kwargs[self.list_field]

class UpdateDictTask(TaskBase):
    """
    A task that updates a dictionary with new key-value pairs.
    """
    def __init__(self, signature: str, **kwargs: Dict[str, Any]):
        super().__init__(signature, **kwargs)
        assert len(self.output_fields) == 1
        self.dict_field = self.output_fields[0]
        self.true_input_fields = self.input_fields
        # add the dict field to the input fields, to ensure the visibility of the dict field
        self.input_fields = self.true_input_fields + [self.dict_field]

    def run(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs[self.dict_field].update(tuple(kwargs[k] for k in self.true_input_fields))
        return kwargs[self.dict_field]

class LLMTask(TaskBase):
    """
    A task that interacts with a language model.
    
    This class provides a framework for preprocessing input data, calling a language model,
    and postprocessing the results.
    
    Attributes:
        params (Dict[str, Any]): Parameters for the language model call.
    """
    
    def __init__(self, signature: str, params: Dict[str, Any], **kwargs: Dict[str, Any]):
        """
        Initialize a language model task.
        
        Args:
            signature (str): The signature defining input and output fields.
            params (Dict[str, Any]): Parameters for the language model call.
        """
        super().__init__(signature, **kwargs)
        self.params = params

    def preprocess(self, **data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Build the messages for the LLM calling.
        
        Args:
            data (Dict[str, Any]): The input data.
            
        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]: A tuple containing:
                - messages: The messages to send to the LLM, or None to skip the LLM call.
                - data: The data to return directly if skipping the LLM call.
                
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Preprocessing is not implemented for this task.")

    def postprocess(self, **data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the result from the LLM.
        
        Args:
            data (Dict[str, Any]): The raw output from the language model.
            
        Returns:
            Dict[str, Any]: The processed output data.
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        return data

    def run(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the language model task with the given input data.
        
        This method handles preprocessing, calling the language model, and postprocessing.
        
        Args:
            data (Dict[str, Any]): The input data.
            
        Returns:
            Dict[str, Any]: The processed output data.
        """
        messages, data = self.preprocess(**kwargs)
        if messages is None:
            assert data is not None
            return data
        else:
            assert data is None
            llm_result = LLMCall().call_llm_multi_thread(messages, self.params, priority=-kwargs.get('@task_step', 0))
            return self.postprocess(**llm_result)

class JSONOutputLLMTask(LLMTask):
    def postprocess(self, **data: Dict[str, Any]) -> Dict[str, Any]:
        # extract the JSON object from the llm result
        reJSON = r"```json\n(.*)\n```"
        match = re.search(reJSON, data['content'], re.S)
        if match is None:
            raise ValueError("No JSON object found in the LLM result.")
        return json.loads(match.group(1))
