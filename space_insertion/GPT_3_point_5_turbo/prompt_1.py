'''
Space insertion using the OpenAI API
'''
import os, sys, yaml, argparse, openai, logging, copy, pathlib

temp = str(pathlib.Path(sys.path[0]).parent.parent)
sys.path.insert(0, temp)
from utils.openai_utils import num_tokens_from_messages, get_completion
from utils.common_utils import cd
sys.path.remove(temp)

CONFIG_DIR = 'configs'
CONFIG_FILE = 'prompt_1.yaml'
openai.api_key = os.getenv("OPENAI_API_KEY")

def user_text_to_prompt(text_lines: list[str], max_tokens: int, line_separator: str = ' ', index_separator: str = ')'
                        ) -> tuple[list[str], set[int]]:
    '''
    Converts the list of user texts into a formatted string for the prompt.
    Each formatted string is made such that the it doesn't exceed the token limit of the model being used.
    Format for the prompt is `'Text lines: 1. text[0]  2. text[1] ...'`
    Returns the list of prompts and the set of indices of text lines which have too many tokens to fit in any prompt.
    '''
    lines_exceeding_tokens = set()
    initial_prompt = 'Text lines:'
    initial_prompt_tokens = num_tokens_from_messages([{'content': initial_prompt}])
    current_tokens = initial_prompt_tokens
    prompts = [initial_prompt]
    prompt_num = 0
    current_lines_in_prompt = []
    i = 0
    while i < len(text_lines):
        prompt_for_line = line_separator + str(i+1) + index_separator + ' ' + text_lines[i]
        tokens_in_line = num_tokens_from_messages([{'content': prompt_for_line}])
        if current_tokens \
        + tokens_in_line \
        + num_tokens_in_response(current_lines_in_prompt, line_separator, index_separator) \
        <= max_tokens:
            prompts[prompt_num] = prompts[prompt_num] + prompt_for_line
            current_tokens = current_tokens + tokens_in_line
            current_lines_in_prompt.append(text_lines[i])
        else:
            if initial_prompt_tokens \
            + tokens_in_line \
            + num_tokens_in_response([text_lines[i]], line_separator, index_separator) \
            > max_tokens:
                logging.warning(
                    'Too many tokens in text line. Continuing without adding spaces.\n'+
                    'Text line: ' + str(text_lines[i])
                    )
                lines_exceeding_tokens.add(i)
            else:
                prompt_num = prompt_num + 1
                prompts.append(initial_prompt + prompt_for_line)
                current_tokens = initial_prompt_tokens + tokens_in_line
                current_lines_in_prompt = [text_lines[i]]
        i = i + 1

    return prompts, lines_exceeding_tokens

def num_tokens_in_response(current_lines: list[str], line_separator: str, index_separator: str) -> int:
    '''
    Estimates the number of tokens in the response which would be produced for an input with `current_lines` as the 
    text lines
    '''
    if len(current_lines)==0:
        return 0
    dummy_str = ''
    for i in range(len(current_lines)):
        dummy_str = dummy_str + str(i+1) + index_separator + ' '+ current_lines[i] + line_separator
    return num_tokens_from_messages([{'content': dummy_str}])

def response_to_list(response: str) -> list[str]:
    '''
    Converts a string of the form `'1. <1st text line> 2. <2nd text line> ...'` into a list containing the text lines.
    '''
    result = response.split('\n')
    temp = []
    for text_line in result:
        text_line = text_line.rstrip().lstrip()
        if len(text_line)==0:
            continue
        # Now we remove the numbering before each text line
        i = 0
        while i<len(text_line) and text_line[i].isnumeric():
            i = i+1
        i = i+1 # For period after numbering
        while i<len(text_line) and text_line[i].isspace():
            i = i+1
        temp.append(text_line[i:])
    return temp

def insert_spaces(text_lines: list[str], 
                  args: dict[str]={
                      'config_dir': CONFIG_DIR,
                      'config_file': CONFIG_FILE
                  },
                  config_path='.'):
    '''
    Inserts spaces in each text line in `text_lines`.
    Result is affected both by the text given and system prompt that has been set.

    Positional Arguments:
        - text_lines (list[str]): Text lines into which spaces are inserted
        - args (dict[str]): Arguments `config_dir` and `config_file` which specify path of config
    '''
    with cd(config_path):
        with open(os.path.join(args['config_dir'], args['config_file']), 'r') as config_fp:
            config = yaml.safe_load(config_fp)
    # `max_tokens_in_user_response` can be 2 * max_tokens_in_user_input in the worst case when there needs to be a
    # space inserted between every two characters. So the remaining tokens are divided as 1/3 for user input and 2/3
    # for response.
    max_tokens_in_user_input = (config['model']['max_tokens'] - num_tokens_from_messages(config['prompts']))//3
    max_tokens_in_response = max_tokens_in_user_input * 2
    user_prompts, lines_exceeding_tokens = user_text_to_prompt(
        text_lines, 
        max_tokens_in_user_input, 
        config['separators']['line_sep'],
        config['separators']['index_sep'])
    global USER_PROMPT
    USER_PROMPT = str(user_prompts)
    lines_with_spaces = []
    i = 0
    for user_prompt in user_prompts:
        prompts = copy.deepcopy(config['prompts'])
        prompts[-1]['content'] = prompts[-1]['content'] + config['separators']['prompt_sep'] + user_prompt
        response = get_completion(config, prompts)
        result = response_to_list(response['choices'][0]['message']['content'])
        if len(lines_exceeding_tokens)+len(result)>len(text_lines):
            result = result[:len(text_lines)-len(lines_exceeding_tokens)]
            logging.warning(
                'Response had more text lines than input so it was truncated to maintain the following condition '+
                '\nnumber of lines exceeding token limit + number of lines in response = number of input lines\n'+
                'The actual response for some lines may have been lost in the truncation.'
            )

        j = 0
        while j < len(result):
            if i in lines_exceeding_tokens:
                lines_with_spaces.append(text_lines[i])
            else:
                lines_with_spaces.append(result[j])
                j = j+1
            i = i + 1
    if len(lines_with_spaces)<len(text_lines):
        logging.warning(
            'Unable to convert entire response to text lines with spaces. Continuing without adding spaces for the '+
            'last `n` lines where `n` is the difference between the number of lines in the converted response and '+
            'the number of lines in the input. Note that the last `n` lines may not be the lines for which no '+
            'response/an unconvertible response was given. Instead responses for two lines may have been merged or a '+
            'response may have been corrupted for some line in the middle. The action taken here is a last-resort '+
            'action and may result in a mismatch between lines in the input and lines returned since the last `n` '+
            'responses are naively copied to the output as given in the input in order to match the number of lines '+
            'in the input and output.\nResponse received:\n'+response['choices'][0]['message']['content'])
        
        while len(lines_with_spaces)<len(text_lines):
            lines_with_spaces.append(text_lines[i])
            i = i + 1

    return lines_with_spaces

def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_dir', help='Directory containing configuration files', default=CONFIG_DIR)
    arg_parser.add_argument('--config_file', help='Name of configuration file including extension', default=CONFIG_FILE)
    args = arg_parser.parse_args()
    args = vars(args)
    with open(os.path.join(args['config_dir'], args['config_file']), 'r') as config_fp:
        config = yaml.safe_load(config_fp)
    with open(config['input']['file'], 'r') as input_fp:
        text_lines = [line.rstrip('\n') for line in input_fp.readlines()]
    response = insert_spaces(text_lines, args)
    for idx, line in enumerate(response):
        print(str(idx+1)+'. '+line)
    if response!=-1:
        with open(os.path.join(config['logs']['dir'] , config['logs']['file']), 'a') as logfile:
            logfile.write('\nPrompt\n')
            for message in config['prompts']:
                logfile.write(message['role']+': '+message['content']+'\n')
            logfile.write('Input: '+USER_PROMPT+'\nOutput: '+str(response)+'\n')

if __name__=='__main__':
    main()