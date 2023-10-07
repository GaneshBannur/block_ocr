import numpy as np
from collections import defaultdict, Counter
from copy import deepcopy

def read_top_to_bottom(bbox_1: np.ndarray, bbox_2: np.ndarray):
  '''
  Determine whether the given bounding boxes would have lines which are read from top to bottom
  (the alternative would be left to right).
  This equates to determining whether two bounding boxes are aligned more along the x-axis or y-axis.
  '''
  # If there was a simultaneous minmax function in numpy we could've used it :(
  max_1 = np.max(bbox_1, axis=0)
  max_2 = np.max(bbox_2, axis=0)
  min_1 = np.min(bbox_1, axis=0)
  min_2 = np.min(bbox_2, axis=0)
  overlap = np.minimum(max_1, max_2) - np.maximum(min_1, min_2)
  bbox_1_dim = max_1 - min_1
  bbox_2_dim = max_2 - min_2
  max_dim = np.maximum(bbox_2_dim, bbox_1_dim)
  min_dim = np.minimum(bbox_2_dim, bbox_1_dim)
  # divide by the max dimension in x and y axis to get the overlap as a fraction of the dimensions of one of the
  # bounding boxes
  max_normalized_overlap = np.divide(overlap, max_dim)
  min_normalized_overlap = np.divide(overlap, min_dim)
  return max_normalized_overlap, min_normalized_overlap

def group_same_level_paras(
        bboxes: list[np.ndarray], 
        overlaps: defaultdict, 
        x_alignments: defaultdict, 
        alignment_axis: str):
    
    if alignment_axis.casefold()=='x':
        grouping_index = 1
        grouping_bool = True
    elif alignment_axis.casefold()=='y':
        grouping_index = 0
        grouping_bool = False
    else:
        print(f'Error: Argument `alignment_axis` must be either \'x\' or \'y\'. Received {alignment_axis}')
        return
    grouped_paras = [{i} for i in range(0, len(bboxes))]
    for i in range(0, len(bboxes)):
        max_overlap = 0
        max_overlap_idx = None
        for idx, overlap in overlaps[i].items():
            if x_alignments[i][idx]==grouping_bool:
                continue
            if overlap[grouping_index]>=max_overlap:
                max_overlap = overlap[grouping_index]
                max_overlap_idx = idx
        if max_overlap<0.5:
            continue
        i_group = None
        max_group = None
        for group_idx, group in enumerate(grouped_paras):
            if i in group and max_overlap_idx in group:
                break
            elif i in group:
                i_group = group_idx
            elif max_overlap_idx in group:
                max_group = group_idx
        if i_group is None:
            continue
        grouped_paras[i_group] = grouped_paras[i_group].union(grouped_paras[max_group])
        grouped_paras.pop(max_group)
    return grouped_paras
  
def para_read_top_to_bottom(bboxes: list[np.ndarray]) -> bool:
    '''
    Determine whether the lines in the given para should be read top to bottom (as opposed to left to right).
    This equates to determining whether the bounding boxes are aligned more along the x-axis or y-axis.
    '''
    x_alignments = defaultdict(lambda: {})
    overlaps = defaultdict(lambda: {})
    flat_list = []
    for i in range(0, len(bboxes)):
        for j in range(i+1, len(bboxes)):
            overlap, min_norm_overlap = read_top_to_bottom(bboxes[i], bboxes[j])
            overlaps[i][j] = min_norm_overlap
            x_alignments[i][j] = True if overlap[0]>overlap[1] else False
            flat_list.append(True if overlap[0]>overlap[1] else False)

    counts = Counter(flat_list)
    
    if counts[True]>counts[False]:
        alignment_axis = 'x'              
        res = True
    else:
        alignment_axis = 'y'
        res = False
    grouped_paras = group_same_level_paras(bboxes, overlaps, x_alignments, alignment_axis)
    return res, grouped_paras

def flatten_list(l:list):
    return [item for sublist in l for item in sublist]

def all_possible_reading_orders(paragraphs: defaultdict[dict[str, list]], image_coordinates):
    '''
    Narrows down the reading order of texts in a paragraph into at most 4 options. 
    
    Narrowing down is done based on the positions of the bounding boxes of the paras. Bounding boxes are assumed to be
    from the coordinate system where x-axis increases towards the right and y-axis increases towards the bottom if
    `image_coordinates` is `True`
    
    Each paragraph in `paragraphs` must be given in the following form:  

    ```python
    {
        'bboxes': list[np.array]
        'texts': list[str]
    }
    ```
    
    Returns a dict for each para of the form (if a reading order doesn't exist it is given as `None`):  

    For paras read top-to-bottom
    ```python
    {
        'top-to-bottom': {
            'left-to-right': {
                'texts': <list of strings containing lines read top-to-bottom and left-to-right>
                'bboxes': <list of bboxes corressponding to the strings>
            }
            'right-to-left': {
                'texts': <list of strings containing lines read top-to-bottom and right-to-left>
                'bboxes': <same as above>
            } 
        }
        'bottom-to-top': {
            'left-to-right': {
                'texts': <list of strings containing lines read top-to-bottom and left-to-right>
                'bboxes': <same as above>
            }   
            'right-to-left': {...}
        }
    }  
    ```
    For paras read left-to-right
    ```python
    {
        'left-to-right': {
            'top-to-bottom': {...} 
            'bottom-to-top': {...}  
        }
        'right-to-left': {
            'top-to-bottom': {...} 
            'bottom-to-top': {...}  
        }
    }
    ```
    '''
    processed_paras = deepcopy(paragraphs) # The original bounding boxes in the paras will be used to get return values
    for para in processed_paras.values():
        para['bboxes'] = np.array(para['bboxes'])
        if not image_coordinates:
            # Flip Y coordinates as the logic implemented here is for image coordinates
            para['bboxes'][:, :, 1] = np.max(para['bboxes'][:, :, 1]) - para['bboxes'][:, :, 1]

    order_strings = {0: 'left-to-right', 1: 'top-to-bottom', 2: 'right-to-left', 3:'bottom-to-top'}
    reading_orders = defaultdict(lambda: {})
    for para_id, para in processed_paras.items():
        top_to_bottom, grouped_paras = para_read_top_to_bottom(para['bboxes'])
        if top_to_bottom:
           sorting_axis = 1                 # sort along y axis for top to bottom reading
        else:
           sorting_axis = 0                 # sort along x axis for left to right reading
        group_sorting_axis = 1 - sorting_axis
        groups_sorted = []
        sorting_coordinate_of_group = []
        for group in grouped_paras:
            sorted_group = sorted(group, key=lambda idx: np.min(para['bboxes'][idx], axis=0)[group_sorting_axis])
            groups_sorted.append(sorted_group)
            group_min_coordinate_idx = min(group, key=lambda idx: np.min(para['bboxes'][idx], axis=0)[sorting_axis])
            sorting_coordinate_of_group.append(np.min(para['bboxes'][group_min_coordinate_idx], axis=0)[sorting_axis])
        group_order = sorted(range(len(groups_sorted)), key=lambda idx: sorting_coordinate_of_group[idx])
        ordered_groups = []
        for idx in group_order:
            ordered_groups.append(groups_sorted[idx])

        # default reading order is top-to-bottom-left-to-right for paras read top-to-bottom and 
        # left-to-right-top-to-bottom for paras read left-to-right
        default_reading_order = {'texts': [], 'bboxes': []}
        for group in ordered_groups:
            default_reading_order['texts'].append([])
            default_reading_order['bboxes'].append([])
            for idx in group:
                default_reading_order['texts'][-1].append(paragraphs[para_id]['texts'][idx])
                default_reading_order['bboxes'][-1].append(paragraphs[para_id]['bboxes'][idx])
        
        reverse_whole = True if len(default_reading_order['texts'])>1 else False
        reverse_groups = False
        for group in default_reading_order['texts']:
            if len(group)>1:
                reverse_groups = True
                break
        
        temp = defaultdict(lambda: {})
        temp[order_strings[sorting_axis]][order_strings[group_sorting_axis]] = default_reading_order
        if reverse_whole:
            temp[order_strings[2+sorting_axis]][order_strings[group_sorting_axis]] = {
                'texts': list(reversed(default_reading_order['texts'])),
                'bboxes': list(reversed(default_reading_order['bboxes']))
            }
        else:
            temp[order_strings[2+sorting_axis]][order_strings[group_sorting_axis]] = None
        if reverse_groups:
            temp[order_strings[sorting_axis]][order_strings[2+group_sorting_axis]] = {
                'texts': [list(reversed(group)) for group in default_reading_order['texts']],
                'bboxes': [list(reversed(group)) for group in default_reading_order['bboxes']]
            }
        else:
            temp[order_strings[sorting_axis]][order_strings[2+group_sorting_axis]] = None
        if reverse_whole and reverse_groups:
            temp[order_strings[2+sorting_axis]][order_strings[2+group_sorting_axis]] = {
                'texts': [list(reversed(group)) for group in reversed(default_reading_order['texts'])],
                'bboxes': [list(reversed(group)) for group in reversed(default_reading_order['bboxes'])]
            }
        else:
            temp[order_strings[2+sorting_axis]][order_strings[2+group_sorting_axis]] = None
        for primary_order_string, secondary_orders in temp.items():
            reading_orders[para_id][primary_order_string] = {}
            for secondary_order_string, order in secondary_orders.items():
                reading_orders[para_id][primary_order_string][secondary_order_string] = {
                    'texts': flatten_list(order['texts']),
                    'bboxes': np.array(flatten_list(order['bboxes']))
                } if order is not None else None

    return dict(reading_orders)
