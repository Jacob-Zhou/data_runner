import asyncio
from typing import Any, Dict, List

from data_runer.tasks import TaskBase

class Pipeline:
    def __init__(self, task: TaskBase):
        self.task = task

    async def __batch_async_call(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronously batch call the task with the given input data.
        """
        return await asyncio.gather(*[self.task.async_call(**d) for d in batch_data])
    
    def batch_call(self, batch_data: List[Dict[str, Any]], async_call: bool = True) -> List[Dict[str, Any]]:
        """
        Batch call the task with the given input data.
        """
        if async_call:
            return asyncio.run(self.__batch_async_call(batch_data))
        else:
            return [self.task(**d) for d in batch_data]

