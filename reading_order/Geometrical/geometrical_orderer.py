import sys, pathlib

temp = str(pathlib.Path(sys.path[0]).parent)
sys.path.insert(0, temp)
from four_orderer import all_possible_reading_orders
sys.path.remove(temp)

def decide_reading_order(paras, args=None, config_path = '.'):
    sentences = []
    paras_dict = {}
    for idx, para in enumerate(paras):
        paras_dict[idx] = {
            'bboxes': para['bboxes'],
            'texts': para['line_texts']
        }
    all_orders = all_possible_reading_orders(paras_dict, image_coordinates=True)
    for order in all_orders.values():
        if order.get('top-to-bottom') is not None:
            sentences.append(' '.join(order['top-to-bottom']['left-to-right']['texts']))
        else:
            sentences.append(' '.join(order['left-to-right']['top-to-bottom']['texts']))
    return sentences
