import tiktoken, openai
from tenacity import (
    retry,
    wait_random_exponential
)

def num_tokens_from_messages(messages: list[dict[str, str]], model="gpt-3.5-turbo-0301"):
    """
    Returns the number of tokens used by a list of messages. 
    Assumes gpt-3.5-turbo-0301 is the model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
                                    See https://github.com/openai/openai-python/blob/main/chatml.md for information 
                                    on how messages are converted to tokens.""")
    
@retry(wait=wait_random_exponential(min=1, max=60))
def get_completion(config, prompts, max_tokens=None):
    try:
        return openai.ChatCompletion.create(
                    model=config['model']['name'],
                    messages = prompts,
                    temperature=config['model']['temperature'],
                    max_tokens=max_tokens
            )
    except Exception as e:
        print(e)
        raise e
