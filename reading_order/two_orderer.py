import numpy as np
from collections import Counter

def read_top_to_bottom(bbox_1: np.ndarray, bbox_2: np.ndarray, bbox_1_dim: tuple[float, float], 
                       bbox_2_dim: tuple[float, float]) -> bool:
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
  max_dim = np.maximum(np.asarray(bbox_2_dim), np.asarray(bbox_1_dim))
  # divide by the max dimension in x and y axis to get the overlap as a fraction of the dimensions of one of the bounding boxes
  overlap = np.divide(overlap, max_dim)
  if overlap[0]>overlap[1]:
     return True
  else:
     return False
  
def para_read_top_to_bottom(bboxes: list[np.ndarray], dims: list[tuple]) -> bool:
    '''
    Determine whether the lines in the given para should be read top to bottom (as opposed to left to right).
    This equates to determining whether the bounding boxes are aligned more along the x-axis or y-axis.
    '''
    alignments = []
    for i in range(0, len(bboxes)):
        for j in range(i+1, len(bboxes)):
            alignments.append(read_top_to_bottom(bboxes[i], bboxes[j], dims[i], dims[j]))
    counts = Counter(alignments)
    if counts[True]>counts[False]:
       return True
    else:
       return False

def narrow_down_reading_order(paras: list[dict[str, list]]):
    '''
    Narrows down the reading order of lines in a paragraph into 2 options. Narrowing down is done based on the 
    positions of the bounding boxes of the lines. Bounding boxes are assumed to be from the coordinate system where 
    x-axis increases towards the right and y-axis increases towards the bottom. Returns the two possible reading orders.
    '''
    reading_orders = []
    order_strings = {0: 'left-to-right', 1: 'top-to-bottom', 2: 'right-to-left', 3: 'bottom-to-top'}
    for para in paras:
        if para_read_top_to_bottom(para['bboxes'], para['bbox_dims']):
           sorting_axis = 1                 # sort along y axis for top to bottom reading
        else:
           sorting_axis = 0                 # sort along x axis for left to right reading
        one_possible_reading_order = [line_text for _, line_text in sorted(
                                        zip(para['bboxes'], para['line_texts']),
                                        key=lambda bbox_plus_id: np.min(bbox_plus_id[0], axis=0)[sorting_axis])]
        reading_orders.append({
           order_strings[sorting_axis]: one_possible_reading_order,
           order_strings[sorting_axis+2]: list(reversed(one_possible_reading_order)),
           order_strings[1-sorting_axis]: None,
           order_strings[3-sorting_axis]: None
           })
    return reading_orders
