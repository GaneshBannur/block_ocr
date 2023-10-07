import sys, os, argparse, cv2, yaml
import tensorflow as tf
from absl import logging
import numpy as np

CONFIG_DIR = 'configs'
CONFIG_FILE = 'config.yaml'

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gin_file', required=True, help='Path to the Gin file that defines the model.')
arg_parser.add_argument('--ckpt_path', required=True, help='Path to the checkpoint directory.')
arg_parser.add_argument('--img_size', default=1024, help='Size of the image fed to the model.')
arg_parser.add_argument('--vis_dir', help='Path for the visualization output.')
arg_parser.add_argument('--output_path', help='Path for the output.')
arg_parser.add_argument('--config_dir', help='Directory containing configuration files', default=CONFIG_DIR)
arg_parser.add_argument('--config_file', help='Name of configuration file including extension', default=CONFIG_FILE)
input_group = arg_parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--img_file', help='Paths to the images.')
input_group.add_argument('--webcam', action='store_true', default=False, help='Flag to use webcam as input')
input_group.add_argument('--vid_file', default=[], help='Path to video file')
input_group.add_argument('--img_dir', help='Paths to the image directories.')

args = arg_parser.parse_args()
if (args.img_file is not None or args.img_dir is not None) and args.output_path is None:
    arg_parser.error('the following arguments are required: output_path')
if (args.img_file is not None or args.img_dir is not None) and args.vis_dir is None:
    arg_parser.error('the following arguments are required: vis_dir')

args_dict = vars(args)

if args.img_file is not None or args.img_dir is not None:
    args_dict['vid'] = False
else:
    args_dict['vid'] = True
    if args_dict['webcam']:
        args_dict['vid_file'] = 0

with open(os.path.join(args_dict['config_dir'], args_dict['config_file']), 'r') as config_fp:
    config = yaml.safe_load(config_fp)

sys.path.insert(0, config['detection']['path'])
from unified_detector.models.official.projects.unified_detector.single_image_detection \
    import load_model as load_detection_model, \
        initialize_flags as initialize_detector_flags, \
        detect_func as detect_text
sys.path.remove(config['detection']['path'])

sys.path.insert(0, config['recognition']['path'])
from parseq.parseq.single_image_inference \
    import recognize_text, \
            load_model_and_transforms as load_recognition_model_and_transforms
sys.path.remove(config['recognition']['path'])

sys.path.insert(0, config['reading_order']['path'])
from reading_order.Geometrical.geometrical_orderer \
     import decide_reading_order
sys.path.remove(config['reading_order']['path'])

sys.path.insert(0, config['space_insertion']['path'])
from space_insertion.GPT_3_point_5_turbo.prompt_1 \
     import insert_spaces
sys.path.remove(config['space_insertion']['path'])

if os.getenv("OPENAI_API_KEY") is None:
    raise SystemExit("Error: Set OPENAI_API_KEY")

initialize_detector_flags(args_dict)

def get_imgs():
    img_lists = []
    if args_dict['img_file'] is not None:
        img_lists.append(args_dict['img_file'])
    if args_dict['img_dir'] is not None:
        img_lists.extend(tf.io.gfile.glob(os.path.join(args_dict['img_dir'], '*')))

    logging.info('Total number of input images: %d', len(img_lists))
    return img_lists

def get_bottom_left(bboxes: list[np.ndarray]):
    '''
    Bounding boxes are assumed to be from the coordinate system where x-axis increases towards the right and y-axis
    increases towards the bottom.
    '''
    bottom_lefts = []
    for bbox in bboxes:
        left = np.min(bbox, axis=0)[0]
        bottom = np.max(bbox, axis=0)[1]
        bottom_lefts.append((left, bottom))
    
    bottom_left = (min(bottom_lefts, key=lambda point: point[0])[0], max(bottom_lefts, key=lambda point: point[1])[1])
    return bottom_left

def get_extremes(bboxes: list[np.ndarray]) -> dict[str, int]:
    '''
    Gets the bottom, top, left and right coordinates of a paragraph.
    The dictionary returned uses `bottom`, `top`, `left` and `right` as keys.
    Bounding boxes are assumed to be from the coordinate system where x-axis increases towards the right and y-axis
    increases towards the bottom.
    '''
    bottoms = []
    tops = []
    lefts = []
    rights = []
    for bbox in bboxes:
        left, top = np.min(bbox, axis=0)
        right, bottom = np.max(bbox, axis=0)
        bottoms.append(bottom)
        tops.append(top)
        lefts.append(left)
        rights.append(right)
    para_bottom = max(bottoms)
    para_top = min(tops)
    para_left = min(lefts)
    para_right = max(rights)
    return {'bottom': para_bottom,
            'top': para_top,
            'left': para_left,
            'right': para_right}

def add_recognized_texts(paras, line_texts):
    for para in paras:
        para['line_texts'] = []
        for idx in para['line_ids']:
            para['line_texts'].append(line_texts[idx])
    return paras

def get_bottom_lefts_of_paras(paras: list[dict[str, list]]):
    bottom_lefts = []
    for para in paras:
        bottom_lefts.append(get_bottom_left(para['bboxes']))
    return bottom_lefts

def visualize_paras(img: np.ndarray, paras: list[dict[str, list]]) -> np.ndarray:
    for para in paras:
        para_extremes = get_extremes(para['bboxes'])
        para_bbox = np.array([[para_extremes['left'], para_extremes['bottom']],
                              [para_extremes['left'], para_extremes['top']],
                              [para_extremes['right'], para_extremes['top']],
                              [para_extremes['right'], para_extremes['bottom']]])
        cv2.drawContours(img, [para_bbox], 0, (0, 0, 255), 2)
    return img


def detect_image(detection_model, recognition_model, recognition_transforms, img_lists):
    detected_imgs, full_img, paras = detect_text(detection_model, img_lists)
    if len(detected_imgs)==0:
        return []
    line_texts, confidences = recognize_text(recognition_model, recognition_transforms, detected_imgs)
    '''lines_with_spaces = insert_spaces(line_texts, config_path=config['space_insertion']['path'])
    line_texts = lines_with_spaces'''
    paras = add_recognized_texts(paras, line_texts)

    one_line_paras_removed = []
    one_line_para_indices = set()
    for idx, para in enumerate(paras):
        if len(para['line_texts'])==1:
            one_line_para_indices.add(idx)
        else:
            one_line_paras_removed.append(para)
        
    multiline_sentences = decide_reading_order(one_line_paras_removed, config_path=config['reading_order']['path'])
    
    multiline_idx = 0
    for idx, para in enumerate(paras):
        if idx in one_line_para_indices:
            para['para_text'] = para['line_texts'][0]
        else:
            para['para_text'] = multiline_sentences[multiline_idx]
            multiline_idx = multiline_idx + 1

    if config['visualize']['paras']:
        full_img = visualize_paras(full_img, paras)
    if config['visualize']['text']:
        bottom_left_of_paras = get_bottom_lefts_of_paras(paras)
        print('Texts')
        for idx, para in enumerate(paras):
            full_img = cv2.circle(full_img, 
                                bottom_left_of_paras[idx],
                                radius=10,
                                color=(0, 0, 255),
                                thickness=-1)
            cv2.putText(img=full_img,
                        text=str(idx+1),
                        org=bottom_left_of_paras[idx],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(255, 0, 0),
                        thickness=3,
                        lineType=2)
            print(f'''{idx+1}. {para['para_text']}''')
    if config['save']:
        key = args_dict['img_file'].split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(args_dict['vis_dir'], f'{key}-detected.jpg'), cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR))
        for idx, detected_img in enumerate(detected_imgs):
            cv2.imwrite(os.path.join(args_dict['vis_dir'], f'{key}-{idx}.jpg'), cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR))

    return paras

def detect_video(detection_model, recognition_model, recognition_transforms, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==False:
            print('Can\'t receive frame, stream may have ended')
            break
        detected_imgs, _, bottom_left_of_boxes = detect_text(detection_model, [frame])
        
        i=0
        for img in detected_imgs:
            label, confidences = recognize_text(recognition_model, recognition_transforms, img)
            cv2.putText(img=frame,
                        text=label,
                        org=bottom_left_of_boxes[i],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=2,
                        lineType=2)
            i = i+1
        
        cv2.imshow('ocr', frame)

        if cv2.waitKey(1)==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    if args_dict['img_file'] is not None or args_dict['img_dir'] is not None:
        img_lists = get_imgs()
    else:
        cap = cv2.VideoCapture(args_dict['vid_file'])

    initialize_detector_flags(args_dict)
    detection_model = load_detection_model()
    recognition_model, recognition_transforms = load_recognition_model_and_transforms()

    if args_dict['img_file'] is not None or args_dict['img_dir'] is not None:
        detect_image(detection_model, recognition_model, recognition_transforms, img_lists)
    else:
        detect_video(detection_model, recognition_model, recognition_transforms, cap)

if __name__=='__main__':
    main()
