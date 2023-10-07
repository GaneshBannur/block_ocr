'''
Line ordering using GPT-3.5-turbo
'''
import openai, os, yaml, argparse, logging, copy, json, sys, pathlib
import numpy as np

temp = str(pathlib.Path(sys.path[0]).parent)
sys.path.insert(0, temp)
from four_orderer import all_possible_reading_orders
sys.path.remove(temp)

temp = str(pathlib.Path(sys.path[0]).parent.parent)
sys.path.insert(0, temp)
from utils.openai_utils import num_tokens_from_messages, get_completion
from utils.common_utils import cd
sys.path.remove(temp)

CONFIG_DIR = 'configs'
CONFIG_FILE = 'prompt_2.yaml'
openai.api_key = os.getenv("OPENAI_API_KEY")

def user_text_to_prompt(paras: list):
    """
    Converts the text lines in the para given into the format required by the prompt, which is:  
    ```python
    your_task(  
        String: '''<text of line 1>''' Bounding Box: [[1, 2] [3, 4] [5, 6] [7, 8]]  
        String: '''<text of line 2>''' Bounding Box: <bounding box of line 2 in the same format>  
        ...  
    )
    ```
    """
    paras = copy.deepcopy(paras)
    tab = ' '*4
    prompts = []
    paras_dict = {}
    for idx, para in enumerate(paras):
        paras_dict[idx] = {
            'bboxes': np.array(para['bboxes']),
            'texts': para['line_texts']
        }
    all_orders = all_possible_reading_orders(paras_dict, image_coordinates=True)
    for idx, order in all_orders.items():
        if order.get('top-to-bottom') is not None:
            paras[idx] = order['top-to-bottom']['left-to-right']
        else:
            paras[idx] = order['left-to-right']['top-to-bottom']
    for para in paras:
        # Flip Y axis so GPT understands but make sure it's not negative as GPT isn't great with negative
        para['bboxes'][:, :, 1] = np.max(para['bboxes'][:, :, 1]) - para['bboxes'][:, :, 1]
        # Bring X coordinates near 0 as well so the numbers are small
        para['bboxes'][:, :, 0] = para['bboxes'][:, :, 0] - np.min(para['bboxes'][:, :, 0])
        cur_prompt = 'your_task(\n'
        for line_text, bbox in zip(para['texts'], para['bboxes']):
            cur_prompt = f"{cur_prompt}{tab}String: '''{line_text}''' Bounding Box: ["
            point_strs = []
            for point in bbox:
                coord_strs = []
                for coord in point:
                    coord_strs.append(str(np.int0(np.round(coord))))
                point_strs.append(f"[{', '.join(coord_strs)}]")
            cur_prompt = f"{cur_prompt}{' '.join(point_strs)}]\n"
        cur_prompt = cur_prompt + ')\n'
        prompts.append(cur_prompt)
    return prompts

def decide_reading_order(paras: list, 
                         args: dict[str]={
                             'config_dir': CONFIG_DIR,
                             'config_file': CONFIG_FILE
                         },
                         config_path = '.'):
    '''
    The text for each paragraph is returned.
    Result is affected both by the text given and system prompt that has been set.
    '''
    with cd(config_path):
        with open(os.path.join(args['config_dir'], args['config_file']), 'r') as config_fp:
            config = yaml.safe_load(config_fp)
    num_tokens_remaining = config['model']['max_tokens'] - (num_tokens_from_messages(config['prompts']))
    user_prompts = user_text_to_prompt(paras)
    if __name__=='__main__':
        global USER_PROMPTS
        USER_PROMPTS = str(user_prompts)
    sentences = []
    for prompt, para in zip(user_prompts, paras):
        num_prompt_tokens = num_tokens_from_messages([{'content': prompt}])
        # Estimate of number of tokens in response
        num_response_tokens = num_tokens_from_messages([{'content': ' '.join(para['line_texts'])}])
        if num_prompt_tokens + num_response_tokens > num_tokens_remaining:
            # Cannot use GPT so use default reading order
            logging.warning(
                f"Paragraph has more tokens than permitted for {config['model']['name']}. "+
                "Using geometrical order instead."
            )
            all_orders = all_possible_reading_orders({
                0: {
                    'bboxes': para['bboxes'],
                    'texts': para['line_texts']
                    }
                }, image_coordinates=True)[0]
            if all_orders.get('top-to-bottom') is not None:
                sentences.append(' '.join(all_orders['top-to-bottom']['left-to-right']['texts']))
            else:
                sentences.append(' '.join(all_orders['left-to-right']['top-to-bottom']['texts']))
        else:
            prompts = copy.deepcopy(config['prompts'])
            prompts[-1]['content'] = prompts[-1]['content'] + '\n' + prompt
            response = get_completion(config, prompts)
            ordered_response = response['choices'][0]['message']['content']
            original_length = len(' '.join(para['line_texts']))
            # If the response produced is less than half the length or more than double the length then it means that
            # the correct response has not been produced, since more than half of the characters have been modified.
            # In this case we default back to geometrical ordering.
            if len(ordered_response)>=original_length/2 and len(ordered_response)<=2*original_length:
                sentences.append(ordered_response)
            else:
                logging.warning(
                    f"{config['model']['name']} produced a response of length {len(ordered_response)} "+
                    f"but sum of lengths of text lines is {original_length}. The response produced is assumed to "+
                    f"be wrong due to this discrepancy in lengths. "
                    "Using geometrical order instead."
                )
                print(f'''Response: {ordered_response}\nLine Texts: {para['line_texts']}''')
                all_orders = all_possible_reading_orders({
                    0: {
                        'bboxes': para['bboxes'],
                        'texts': para['line_texts']
                        }
                    }, image_coordinates=True)[0]
                if all_orders.get('top-to-bottom') is not None:
                    sentences.append(' '.join(all_orders['top-to-bottom']['left-to-right']['texts']))
                else:
                    sentences.append(' '.join(all_orders['left-to-right']['top-to-bottom']['texts']))
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
        para['bboxes']  = np.array(para['bboxes'])

    response = decide_reading_order(paras, args)
    print('Processed response: ', response)
    if response!=-1:
        with open(os.path.join(config['logs']['dir'] , config['logs']['file']), 'a') as logfile:
            logfile.write('\nTemperature: '+str(config['model']['temperature']))
            logfile.write('\nPrompt\n')
            for message in config['prompts']:
                logfile.write(message['role']+': '+message['content']+'\n')
            logfile.write('Input\n'+USER_PROMPTS+'\nOutput\n'+str(response)+'\n')

if __name__=='__main__':
    main()
