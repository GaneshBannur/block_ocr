'''
Line ordering using GPT-3.5-turbo
'''
import openai, os, yaml, argparse, logging, copy, json, sys, pathlib
import numpy as np

CONFIG_DIR = 'configs'
CONFIG_FILE = 'prompt_1.yaml'
openai.api_key = os.getenv("OPENAI_API_KEY")

temp = str(pathlib.Path(sys.path[0]).parent)
sys.path.insert(0, temp)
from two_orderer import narrow_down_reading_order
sys.path.remove(temp)

temp = str(pathlib.Path(sys.path[0]).parent.parent)
sys.path.insert(0, temp)
from utils.openai_utils import cd, num_tokens_from_messages, get_completion
sys.path.remove(temp)

def user_text_to_prompt(
        text_pairs: list[list[str]], 
        max_tokens: int, 
        line_separator: str = ' ', 
        pair_separator: str = ' ',
        index_separators: list[str] = [')', '.']
        ) -> tuple[list[str], set[int]]:
    '''
    Converts the list of user text pairs into a list of formatted strings for the prompt.
    Each formatted string is made such that the it doesn't exceed the token limit of the model being used.
    Format of each prompt is:
    `'Pairs of text lines: 1) 1. text[0][0] 2. text[0][1]  2) 1. text[1][0] 2. text [1][1] ...'`.
    Returns the list of prompts and the set of indices of text pairs which have too many tokens to fit in any prompt.
    '''
    num_pairs_in_prompt = 0
    pairs_exceeding_tokens = set()
    initial_prompt = 'Pairs of text lines:'
    initial_prompt_tokens = num_tokens_from_messages([{'content': initial_prompt}])
    current_tokens = initial_prompt_tokens
    prompts = [initial_prompt]
    prompt_num = 0
    i = 0
    while i < len(text_pairs):
        prompt_for_pair = line_separator + str(i+1) + index_separators[0]
        for idx, j in enumerate(['1', '2']):
            prompt_for_pair = prompt_for_pair + pair_separator + j + index_separators[1] +' ' + text_pairs[i][idx]
        tokens_in_pair = num_tokens_from_messages([{'content': prompt_for_pair}])
        if current_tokens \
        + tokens_in_pair \
        + num_tokens_in_response(num_pairs_in_prompt, line_separator, index_separators[0]) \
        <= max_tokens:
            prompts[prompt_num] = prompts[prompt_num] + prompt_for_pair
            current_tokens = current_tokens + tokens_in_pair
            num_pairs_in_prompt = num_pairs_in_prompt + 1
        else:
            if initial_prompt_tokens \
            + tokens_in_pair \
            + num_tokens_in_response(1, line_separator, index_separators[0]) \
            > max_tokens:
                logging.warning(
                    'Too many tokens in text pair. Continuing by selecting the first of the pair.\n'+
                    'Text pair: ' + str(text_pairs[i])
                    )
                pairs_exceeding_tokens.add(i)
            else:
                prompt_num = prompt_num + 1
                prompts.append(initial_prompt + prompt_for_pair)
                current_tokens = initial_prompt_tokens + tokens_in_pair
                num_pairs_in_prompt = 1
        i = i + 1

    return prompts, pairs_exceeding_tokens

def response_to_list(response: str, text_pairs: list[list[str]]) -> list[int]:
    '''
    Converts a string of the form `'1. <0 or 1> 2. <0 or 1> ...'` into a list containing indices 0 or 1.  
    '`\\n`', '`\\t`' or '` `' are all considered as spaces.
    Any non-numeric text is ignored and the first number found after each index (`1.`, `2.` , etc) is appended to the
    list which is returned.
    '''
    result = response.split('\n')
    temp = []
    for pair_num, text_line in enumerate(result):
        # ARBITRARY
        if pair_num>=len(text_pairs):
            pair_num = len(text_pairs)-1
        text_line = text_line.rstrip().lstrip()
        if len(text_line)==0:
            continue

        if len(text_line)==1:
            try:
                idx = int(text_line)
            except ValueError:
                idx = 1
                logging.warning(
                'Unable to convert response to index. Continuing by selecting first in the pair.\n'+
                'Text pair: ' + str(text_pairs[pair_num])+'\n'+
                'Response received: ' + text_line)
            temp.append(idx-1)
            continue

        right_end = -len(text_line) # To make `left_end` 0 initially
        while right_end!=-1:
            # while loop removes the numbering before each text line and converts the index to an int
            left_end = len(text_line)+right_end
            # Remove non-numeric characters in the beginning
            while left_end<len(text_line) and not text_line[left_end].isnumeric():
                left_end = left_end + 1
            index_left_end = left_end
            # Remove the number (1., 2), etc)
            while left_end<len(text_line) and text_line[left_end].isnumeric():
                left_end = left_end+1
            if left_end<len(text_line):
                left_end = left_end+1 # For period/bracket/space after numbering
            # Removing extra spaces
            while left_end<len(text_line) and text_line[left_end].isspace():
                left_end = left_end+1
            # If there was only an index then consider the index itself as the ordering
            if left_end>=len(text_line):
                left_end = index_left_end
            
            while left_end<len(text_line):
            # while loop to try with successively incremented `left_end` as the starting point
                try:
                    idx = int(text_line[left_end:])
                    # Set `right_end` to -1 since the entire remaining string was successfully converted to an int
                    right_end = -1
                    break
                except ValueError:
                    right_end = -1
                    while abs(right_end)+left_end<len(text_line):
                        try:
                            idx = int(text_line[left_end:right_end])
                        except ValueError:
                            right_end = right_end - 1
                        else:
                            break
                    else:
                        left_end = left_end + 1
                        continue
                    # Set `right_end` to -1 to break out of main loop
                    right_end = -1
                    break
            else:
                idx = 1
                logging.warning(
                    'Unable to convert response to index. Continuing by selecting first in the pair.\n'+
                    'Text pair: ' + str(text_pairs[pair_num])+'\n'+
                    'Response received: ' + text_line)
            try:
                temp.append(idx-1)
                right_end = -1
            except NameError:
                idx = 1
                temp.append(idx-1)
                logging.warning(
                    'Unable to convert response to index. Continuing by selecting first in the pair.\n'+
                    'Text pair: ' + str(text_pairs[pair_num])+'\n'+
                    'Response received: ' + text_line)
                right_end = -1
    return temp

def num_tokens_in_response(num_text_pairs: int, line_separator: str, index_separator: str) -> int:
    '''
    Estimates the number of tokens in the response which would be produced for an input with `num_text_pairs` text
    pairs
    '''
    if num_text_pairs==0:
        return 0
    dummy_str = ''
    for i in range(num_text_pairs):
        dummy_str = dummy_str + str(i+1) + index_separator +' 1' + line_separator
    return num_tokens_from_messages([{'content': dummy_str}])

def decide_reading_order(paras: list[dict[str, list]], 
                         args: dict[str]={
                             'config_dir': CONFIG_DIR,
                             'config_file': CONFIG_FILE
                         },
                         config_path = '.'):
    '''
    For each para in `paras` the ordered text is returned.
    Result is affected both by the text given and system prompt that has been set.

    Positional Arguments:
        - paras (list[dict[str, list]]): Paragraphs containing `line_texts`, `bboxes`, `bbox_dims`
        - args (dict[str]): Arguments `config_dir` and `config_file` which specify path of config
    '''
    with cd(config_path):
        with open(os.path.join(args['config_dir'], args['config_file']), 'r') as config_fp:
            config = yaml.safe_load(config_fp)
    possible_reading_orders = narrow_down_reading_order(paras)
    text_pairs = []
    for para_orders in possible_reading_orders:
        if para_orders['top-to-bottom'] is not None:
            text_pairs.append([
                para_orders['top-to-bottom'],
                para_orders['bottom-to-top']
            ])
        else:
            text_pairs.append([
                para_orders['left-to-right'],
                para_orders['right-to-left']
            ])
    
    # `max_tokens_in_user_prompt` accounts for the predefined prompt, a space/newline separating predefined and user
    # prompts and also an estimate of how many tokens the response would consume
    max_tokens_in_user_input = config['model']['max_tokens'] - (num_tokens_from_messages(config['prompts'])+1)
    user_prompts, pairs_exceeding_tokens = user_text_to_prompt(
        text_pairs,
        max_tokens_in_user_input,
        config['separators']['line_sep'],
        config['separators']['pair_sep'],
        config['separators']['index_seps'])
    global USER_PROMPT
    USER_PROMPT = str(user_prompts)
    sentences = []
    i = 0
    for user_prompt in user_prompts:
        prompts = copy.deepcopy(config['prompts'])
        prompts[-1]['content'] = prompts[-1]['content'] + config['separators']['prompt_sep'] + user_prompt
        response = get_completion(config, prompts)
        result = response_to_list(response['choices'][0]['message']['content'], text_pairs)
        if len(pairs_exceeding_tokens)+len(result)>len(text_pairs):
            result = result[:len(text_pairs)-len(pairs_exceeding_tokens)]
            logging.warning(
                'Response had more text pairs than input so it was truncated to maintain the following condition '+
                '\nnumber of pairs exceeding token limit + number of pairs in response = number of input pairs\n'+
                'The actual response for some pairs may have been lost in the truncation.'
            )
        j = 0
        while j < len(result):
            if i in pairs_exceeding_tokens:
                sentences.append(text_pairs[i][0])
            else:
                sentences.append(text_pairs[i][result[j]])
                j = j+1
            i = i + 1

    if len(sentences)<len(text_pairs):
        logging.warning(
            'Unable to convert entire response to indices. Continuing by selecting the first in the pair for the '+
            'pairs with no response or response that cannot be decoded.')
        while len(sentences)<len(text_pairs):
            sentences.append(text_pairs[i][0])
            i = i + 1
    return sentences

def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_dir', help='Directory containing configuration files', default=CONFIG_DIR)
    arg_parser.add_argument('--config_file', help='Name of configuration file including extension', default=CONFIG_FILE)
    args = arg_parser.parse_args()
    args = vars(args)
    with open(os.path.join(args['config_dir'], args['config_file']), 'r') as config_fp:
        config = yaml.safe_load(config_fp)

    with open(os.path.join(config['input']['dir'], config['input']['file']), 'r') as input_fp:
        paras = json.load(input_fp)
    for para in paras:
        for bbox in para['bboxes']:
            bbox = np.array(bbox)

    response = decide_reading_order(paras, args)
    print('Response: ', response)
    with open(os.path.join(config['logs']['dir'] , config['logs']['file']), 'a') as logfile:
        logfile.write('\nTemperature: '+str(config['model']['temperature']))
        logfile.write('\nPrompt\n')
        for message in config['prompts']:
            logfile.write(message['role']+': '+message['content']+'\n')
        logfile.write('Input\n'+USER_PROMPT+'\nOutput\n'+str(response)+'\n')

if __name__=='__main__':
    main()