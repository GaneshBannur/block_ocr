import yaml, os, json, statistics, sys
from shapely.geometry import Polygon
from collections import defaultdict
from unidecode import unidecode
from datetime import datetime
from pathlib import Path
import shutil

from utils.common_utils import cd
from reading_order.four_orderer import all_possible_reading_orders
from end_to_end_ocr_manual import (
    detect_image,
    load_detection_model, 
    load_recognition_model_and_transforms
)
from utils.substring_matching import get_best_match

EVALUATION_PATH = 'evaluation'
CONFIG_DIR = 'configs'
CONFIG_FILE = 'manual.yaml'

sys.path.insert(0, os.path.abspath(EVALUATION_PATH))
with cd(EVALUATION_PATH):
    from evaluation.metrics import compute_all_metrics
sys.path.remove(os.path.abspath(EVALUATION_PATH))

with open(os.path.join(EVALUATION_PATH, CONFIG_DIR, CONFIG_FILE), 'r') as config_fp:
    config = yaml.safe_load(config_fp)

config['img_dir'] = os.path.abspath(config['img_dir'])

def save_metrics(res, save_avg=False, save_metrics=False):
    cur_datetime = str(datetime.now())
    metric_names = []
    if save_avg:
            metric_names =  metric_names+ [config['res']['avg']]
    if save_metrics:
            metric_names = metric_names + config['metrics']['names']

    for metric_method in config['metrics']['methods']:
        try:
            Path(os.path.join(
                EVALUATION_PATH, 
                config['res']['dir'], 
                metric_method)
                ).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        
        for metric_name in metric_names:
            try:
                with open(temp_path, 'w') as temp_fp:
                    json.dump({
                            'datetime': cur_datetime,
                            'results': res[metric_method][metric_name]
                        },
                        temp_fp,
                        indent=4)
                shutil.copy2(temp_path, metric_paths[metric_method][metric_name])
            except Exception:
                print(f"Error writing {metric_method}/{metric_name} to file: {temp_path}",
                f"Or error copying {temp_path} to {metric_paths[metric_method][metric_name]}")
                raise SystemExit

def save_preds(preds):
    cur_datetime = str(datetime.now())
    try:
        Path(os.path.join(
            EVALUATION_PATH,
            config['res']['dir']
        )).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    try:
        with open(temp_path, 'w') as temp_fp:
            json.dump({
                    'datetime': cur_datetime,
                    'predictions': preds
                },
                temp_fp,
                indent=4)
        shutil.copy2(temp_path, pred_path)
    except Exception:
        print(f"Error writing predictions to file: {temp_path}",
              f"Or error copying {temp_path} to {pred_path}")
        raise SystemExit
    
def reading_orders_dict_to_list(reading_orders):
    reading_orders_list = defaultdict(lambda: [])
    for group_idx, group_orders in reading_orders.items():
        for primary_reading_order in group_orders:
            for orders in group_orders[primary_reading_order].values():
                if orders is not None:
                    reading_orders_list[group_idx].append(' '.join(orders['texts']))
    return reading_orders_list

metric_paths = {}
for metric_method in config['metrics']['methods']:
    metric_paths[metric_method] = {}
    for metric_name in config['metrics']['names'] + [config['res']['avg']]:
        metric_paths[metric_method][metric_name] = os.path.join(
            EVALUATION_PATH, 
            config['res']['dir'], 
            metric_method,
            metric_name + config['res']['format'])
        
pred_path = os.path.join(EVALUATION_PATH, config['res']['dir'], config['res']['pred']+config['res']['format'])
temp_path = os.path.join(EVALUATION_PATH, config['res']['dir'], config['res']['temp']+config['res']['format'])

for metric_method in config['metrics']['methods']:
    if Path(metric_paths[metric_method][config['res']['avg']]).exists():
        print('Metrics have already been calculated in ' + metric_paths[metric_method]['avg'] + 
            ' Please use a different directoy or move the file to a different directory.')
        raise SystemExit

LOGFILE = os.path.join(EVALUATION_PATH, config['log']['dir'], config['log']['file'])

detection_model = load_detection_model()
recognition_model, recognition_transforms = load_recognition_model_and_transforms()

with open(os.path.join(EVALUATION_PATH, config['gt_file']), 'r') as file:
    annotation = json.load(file)
    annotations = annotation['annotations']

res = {}
predictions = {}

flag = False
for metric_method in config['metrics']['methods']:
    for metric_name in config['metrics']['names']:
        if not Path(metric_paths[metric_method][metric_name]).exists():
            flag = True
            break

if flag==True:
    for metric_method in config['metrics']['methods']:
        res[metric_method] = {}
        for metric_name in config['metrics']['names']:
            res[metric_method][metric_name] = {}
else:
    for metric_method in config['metrics']['methods']:
        res[metric_method] = {}
        for metric_name in config['metrics']['names']:
            with open(metric_paths[metric_method][metric_name], 'r') as metric_fp:
                res[metric_method][metric_name] = json.load(metric_fp)['results']

if Path(pred_path).exists():
    with open(pred_path, 'r') as pred_fp:
        predictions = json.load(pred_fp)['predictions']

for cur_img_idx, cur_img in enumerate(annotations):
    img_filename = cur_img['image_id']

    flag = False
    for metric_method in config['metrics']['methods']:
        for metric_name in config['metrics']['names']:
            if img_filename not in res[metric_method][metric_name]:
                flag = True
                break
    if flag == False:
        continue
    
    for metric_method in config['metrics']['methods']:
        for metric_name in config['metrics']['names']:
            res[metric_method][metric_name][img_filename] = []

    img_lists = [os.path.join(config['img_dir'], img_filename + '.jpg')]

    if img_filename not in predictions:
        pred_paras = detect_image(detection_model,
                                recognition_model,
                                recognition_transforms,
                                img_lists)

        for para in pred_paras:
            para['para_bbox'] = para['para_bbox'].tolist()
            for idx in range(len(para['bboxes'])):
                para['bboxes'][idx] = para['bboxes'][idx].tolist()
        
        predictions[img_filename] = pred_paras
        if cur_img_idx%config['save_frequency']['pred']==0:
            save_preds(predictions)
    else:
        pred_paras = predictions[img_filename]

    temp = []
    for para in pred_paras:
        para['para_text'] = unidecode(para['para_text']).casefold()
        if len(para['para_text'])>0:
            temp.append(para)
    pred_paras = temp

    gt_paras = []
    for para in cur_img['paragraphs']:
        if para.get('text') is not None:
            para['text'] = unidecode(para['text']).casefold()
            if len(para['text'])>0:
                gt_paras.append(para)

    grouped_paras = defaultdict(lambda: {'bboxes': [], 'texts': []})

    for pred_idx, pred_para in enumerate(pred_paras):
        max_iou = 0
        group = -1
        pred_poly = Polygon(pred_para['para_bbox']).buffer(0)
        for gt_idx, gt_para in enumerate(gt_paras):
            gt_poly = Polygon(gt_para['vertices']).buffer(0)
            cur_iou = pred_poly.intersection(gt_poly).area/pred_poly.union(gt_poly).area
            if cur_iou>max_iou:
                group = gt_idx
                max_iou = cur_iou
        if group!=-1:
            grouped_paras[group]['bboxes'].append(pred_para['para_bbox'])
            grouped_paras[group]['texts'].append(pred_para['para_text'])

    best_substring_gts = {}
    for group_id, group in grouped_paras.items():
        group_gt_text = gt_paras[group_id]['text']
        if len(group['texts'])>1:
            best_substring_gts[group_id] = [
                get_best_match(para_text, group_gt_text)['best_match'] for para_text in group['texts']
                ]
        else:
            best_substring_gts[group_id] = [group_gt_text]

    multipara_groups = {}
    single_para_reading_orders = {}
    for group_id, group in grouped_paras.items():
        if len(group['texts'])>1:
            multipara_groups[group_id] = group
        else:
            single_para_reading_orders[group_id] = [group['texts'][0]]
    reading_orders = all_possible_reading_orders(multipara_groups, image_coordinates=True)
    reading_orders = reading_orders_dict_to_list(reading_orders)
    reading_orders = reading_orders | single_para_reading_orders

    for group_id in grouped_paras:
        if (gt_paras[group_id]['considered']==False and 'sensible_order' not in gt_paras[group_id]) \
           or (gt_paras[group_id].get('sensible_order', False)==False):
            continue

        # LCS method
        for pred_string, gt_string in zip(grouped_paras[group_id]['texts'], best_substring_gts[group_id]):
            temp = compute_all_metrics(gt_string, pred_string)

            for metric_name in config['metrics']['names']:
                res['lcs'][metric_name][img_filename].append(temp[metric_name])

        # Reading Order method
        gt_text = gt_paras[group_id]['text']
        possible_perms = reading_orders[group_id]
        possible_perm_metrics = [compute_all_metrics(gt_text, pred_string) for pred_string in possible_perms]

        for metric_name in config['metrics']['names']:
            metric_vals = [m[metric_name] for m in possible_perm_metrics]
            res['reading_order'][metric_name][img_filename].append({
                'min': min(metric_vals),
                'avg': statistics.fmean(metric_vals),
                'max': max(metric_vals)
            })

    with open(LOGFILE, 'a') as logfp:
        cur_datetime = str(datetime.now())
        logfp.write(f"{cur_datetime}: Finished {cur_img_idx+1} images\n")

    if cur_img_idx%config['save_frequency']['metrics']==0:
        save_metrics(res, save_avg=False, save_metrics=True)

save_metrics(res, save_avg=False, save_metrics=True)

for metric_method in config['metrics']['methods']:
    res[metric_method][config['res']['avg']] = {}
    for metric_name in config['metrics']['names']:
        metric_vals = []
        for img_filename in res[metric_method][metric_name]:
            metric_vals.extend(res[metric_method][metric_name][img_filename])
        if metric_method == 'lcs':
            res[metric_method][config['res']['avg']][metric_name] = statistics.fmean(metric_vals)
        if metric_method == 'reading_order':
            grouped_metric_vals = {'min': [], 'avg': [], 'max': []}
            for para_val in metric_vals:
                grouped_metric_vals['min'].append(para_val['min'])
                grouped_metric_vals['avg'].append(para_val['avg'])
                grouped_metric_vals['max'].append(para_val['max'])
            
            res[metric_method][config['res']['avg']][metric_name] = {
                'min': statistics.fmean(grouped_metric_vals['min']),
                'avg': statistics.fmean(grouped_metric_vals['avg']),
                'max': statistics.fmean(grouped_metric_vals['max'])
            }

save_metrics(res, save_avg=True, save_metrics=False)
