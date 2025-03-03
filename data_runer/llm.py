from threading import Lock
import os
import time
from typing import Any, Dict, Iterator, List
from openai import OpenAI
import openai
import httpx
from data_runer.executors import ThreadPoolExecutor

def _stream_call(messages, api_key, base_url, params):
    """
    Make a streaming API call to an OpenAI-compatible endpoint.

    Args:
        messages (list): The list of messages to send to the API.
        api_key (str): The API key to use for the call.
        base_url (str): The base URL to use for the call.
        params (dict): The parameters to use for the call.

    Yields:
        str: The next chunk of the response.
    """
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(http2=True, timeout=10000))
    params.pop('stream', None)
    params.pop('include_usage', None)
    headers = {
        'Accept': 'text/event-stream'
    }
    if 'extra_headers' in params:
        params['extra_headers'].update(headers)
    else:
        params['extra_headers'] = headers

    # only capture the rate limit error
    try:
        response = client.chat.completions.create(
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )

        # for chunk in response:
        #     yield chunk
        return response
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        time.sleep(os.getenv("RATE_LIMIT_WAIT_TIME", 10))
        raise e
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
def _parsing_openai_frame_stream(frame) -> Dict[str, Any]:
    content, reasoning_content, request_id, stop_reason = '', '', '', ''

    request_id = frame.id
    choices = frame.choices
    if len(choices) > 0:
        if choices[0].delta.content is not None:
            content = choices[0].delta.content
        if hasattr(choices[0].delta, 'reasoning_content'):
            reasoning_content = choices[0].delta.reasoning_content
        if reasoning_content is None:
            reasoning_content = ''
        stop_reason = choices[0].finish_reason

    return {
        'content': content,
        'reasoning_content': reasoning_content,
        'request_id': request_id,
        'stop_reason': stop_reason,
    }

def _parse_openai_frame_stream_to_non_stream(frames: Iterator[Any]) -> Dict[str, Any]:
    content, reasoning_content, request_id, stop_reason = '', '', '', ''

    n_frames = 0
    for frame in frames:
        n_frames += 1
        frame_obj = _parsing_openai_frame_stream(frame)
        if os.getenv("NON_INCREMENTAL_FRAME", "false").lower() == "true":
            content = frame_obj['content']
            reasoning_content = frame_obj['reasoning_content']
        else:
            content += frame_obj['content']
            reasoning_content += frame_obj['reasoning_content']
        this_request_id = frame_obj['request_id']
        if request_id == '':
            request_id = this_request_id
        assert this_request_id == request_id, f"request_id is not consistent, need to debug, {this_request_id} != {request_id}"
        stop_reason = frame_obj['stop_reason']

    if n_frames == 0:
        print("No frames received, return empty content")

    return {
        'content': content,
        'reasoning_content': reasoning_content,
        'request_id': request_id,
        'stop_reason': stop_reason,
    }

def call_llm(messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the LLM with the given data.

    Args:
        data (Dict[str, Any]): The data to call the LLM with.

    Returns:
        Dict[str, Any]: The response from the LLM.
    """
    params.pop('stream', None)
    params.pop('include_usage', None)
    api_key = params.pop('api_key', os.getenv("API_KEY"))
    base_url = params.pop('base_url', os.getenv("BASE_URL", "https://api.openai.com/v1"))

    response = _stream_call(messages, api_key, base_url, params)

    return _parse_openai_frame_stream_to_non_stream(response)

def call_llm_stream(messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Call the LLM with the given data.
    """

    params.pop('stream', None)
    params.pop('include_usage', None)
    api_key = params.pop('api_key', os.getenv("API_KEY"))
    base_url = params.pop('base_url', os.getenv("BASE_URL", "https://api.openai.com/v1"))

    response = _stream_call(messages, api_key, base_url, params)

    for frame in response:
        yield _parsing_openai_frame_stream(frame)

class LLMCall:
    """
    A singleton class to manage the LLM call.
    """
    _llm_executor_lock = Lock()
    _llm_executor = {}
    _llm_key_base_url = {}
    _llm_call_object = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._llm_call_object:
            cls._llm_call_object[cls] = super().__new__(cls)
        return cls._llm_call_object[cls]
    
    def __init__(self, *args, **kwargs):
        pass

    def register_model(self, model_name: str, api_key: str, base_url: str, n_workers: int = 1):
        with self._llm_executor_lock:
            if model_name not in self._llm_executor:
                self._llm_executor[model_name] = ThreadPoolExecutor(n_workers)
                self._llm_key_base_url[model_name] = (api_key, base_url)
            else:
                # override the existing executor
                del self._llm_executor[model_name]
                self._llm_executor[model_name] = ThreadPoolExecutor(n_workers)
                self._llm_key_base_url[model_name] = (api_key, base_url)
    
    def call_llm(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        # sync call
        return call_llm({'messages': messages, 'params': params})
    
    def call_llm_multi_thread(self, messages: List[Dict[str, Any]], params: Dict[str, Any], priority: int = 0) -> Dict[str, Any]:
        # async call with priority
        if 'model' not in params:
            raise ValueError("model is not specified")
        params = params.copy()
        model_name = params['model']
        # get the api key and base url from the registered model
        params['api_key'], params['base_url'] = self._llm_key_base_url[model_name]
        if model_name not in self._llm_executor:
            raise ValueError(f"Model {model_name} not registered, it can not be called asynchronously")
        result = self._llm_executor[model_name].submit(call_llm, priority, messages, params)
        result = result.result()
        return result

    def shutdown(self):
        for model_name in self._llm_executor:
            self._llm_executor[model_name].shutdown()

