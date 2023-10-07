# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""A binary to run unified detector."""

import os
from typing import Any, Dict, Sequence, Union

from absl import app
from absl import flags
from absl import logging

import cv2
import gin
import numpy as np
import tensorflow as tf
import tqdm

from official.projects.unified_detector import external_configurables  # pylint: disable=unused-import
from official.projects.unified_detector.modeling import universal_detector
from official.projects.unified_detector.utils import utilities

# group two lines into a paragraph if affinity score higher than this
_PARA_GROUP_THR = 0.5


# MODEL spec
_GIN_FILE = flags.DEFINE_string(
    'gin_file', None, 'Path to the Gin file that defines the model.')
_CKPT_PATH = flags.DEFINE_string(
    'ckpt_path', None, 'Path to the checkpoint directory.')
_IMG_SIZE = flags.DEFINE_integer(
    'img_size', 1024, 'Size of the image fed to the model.')

# Input & Output
# Note that, all images specified by `img_file` and `img_dir` will be processed.
_IMG_FILE = flags.DEFINE_multi_string('img_file', [], 'Paths to the images.')
_IMG_DIR = flags.DEFINE_multi_string(
    'img_dir', [], 'Paths to the image directories.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Path for the output.')
_VIS_DIR = flags.DEFINE_string(
    'vis_dir', None, 'Path for the visualization output.')


def _preprocess(raw_image: np.ndarray) -> Union[np.ndarray, float]:
  """Convert a raw image to properly resized, padded, and normalized ndarray."""
  # (1) convert to tf.Tensor and float32.
  img_tensor = tf.convert_to_tensor(raw_image, dtype=tf.float32)

  # (2) pad to square.
  height, width = img_tensor.shape[:2]
  maximum_side = tf.maximum(height, width)
  height_pad = maximum_side - height
  width_pad = maximum_side - width
  img_tensor = tf.pad(
      img_tensor, [[0, height_pad], [0, width_pad], [0, 0]],
      constant_values=127)
  ratio = maximum_side / _IMG_SIZE.value
  # (3) resize long side to the maximum length.
  img_tensor = tf.image.resize(
      img_tensor, (_IMG_SIZE.value, _IMG_SIZE.value))
  img_tensor = tf.cast(img_tensor, tf.uint8)

  # (4) normalize
  img_tensor = utilities.normalize_image_to_range(img_tensor)

  # (5) Add batch dimension and return as numpy array.
  return tf.expand_dims(img_tensor, 0).numpy(), float(ratio)


def load_model() -> tf.keras.layers.Layer:
  gin.parse_config_file(_GIN_FILE.value)
  model = universal_detector.UniversalDetector()
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_path = _CKPT_PATH.value
  logging.info('Load ckpt from: %s', ckpt_path)
  ckpt.restore(ckpt_path).expect_partial()
  return model


def inference(img_file: Union[str, np.ndarray], model: tf.keras.layers.Layer, vid: bool) -> Dict[str, Any]:
  """Inference step."""
  if not vid:
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
  else:
    img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
  img_ndarray, ratio = _preprocess(img)

  output_dict = model.serve(img_ndarray)
  
  class_tensor = output_dict['classes'].numpy()
  mask_tensor = output_dict['masks'].numpy()
  group_tensor = output_dict['groups'].numpy()

  indices = np.where(class_tensor[0])[0].tolist()  # indices of positive slots.
  mask_list = [
      mask_tensor[0, :, :, index] for index in indices]  # List of mask ndarray.

  '''Uncomment line below to visualize line masks'''
  #visualize_masks(mask_list, indices)

  # Form lines and words
  lines = []
  line_indices = []
  for index, mask in tqdm.tqdm(zip(indices, mask_list)):
    line = {
        'words': [],
        'text': '',
    }

    contours, _ = cv2.findContours(
        (mask > 0.).astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]

    '''
    Merging all the pseudo-word-level contours of a line into groups of line-level contour.
    If contours are very far apart they are kept in separate groups even though they are part of the same line.
    '''
    num_points = 0
    num_groups = 0
    cnt_groups = []
    for idx, contour in enumerate(contours):
      if (isinstance(contour, np.ndarray) and
          len(contour.shape) == 3 and
          contour.shape[0] > 2 and
          contour.shape[1] == 1 and
          contour.shape[2] == 2):
        num_points = num_points + contour.shape[0]
        for group in cnt_groups:
          for grouped_cnt in group:
            if is_same_group(contour, contours[grouped_cnt]):
              group.append(idx)
              break
          else:
            continue
          break
        else:
          cnt_groups.append([idx])
          num_groups = num_groups + 1

      else:
        logging.error('Invalid contour: %s, discarded', str(contour))
    
    merged_cnt = np.empty((num_groups, num_points, 1, 2), np.int32)
    for group_num, group in enumerate(cnt_groups):
      valid_idx = 0
      for cnt_idx in group:
        contour = contours[cnt_idx]
        merged_cnt[group_num, valid_idx: valid_idx+contour.shape[0]] = contour
        valid_idx = valid_idx + contour.shape[0]

      cnt_list = (merged_cnt[group_num, 0:valid_idx, 0] * ratio).astype(np.int32).tolist()
      line['words'].append({'text': '', 'vertices': cnt_list})
    '''End of merging all the pseudo-word-level contours of a line into one single line-level contour'''

    '''for contour in contours:
      if (isinstance(contour, np.ndarray) and
          len(contour.shape) == 3 and
          contour.shape[0] > 2 and
          contour.shape[1] == 1 and
          contour.shape[2] == 2):
        cnt_list = (contour[:, 0] * ratio).astype(np.int32).tolist()
        line['words'].append({'text': '', 'vertices': cnt_list})
      else:
        logging.error('Invalid contour: %s, discarded', str(contour))'''

    if line['words']:
      lines.append(line)
      line_indices.append(index)

  # Form paragraphs
  line_grouping = utilities.DisjointSet(len(line_indices))
  affinity = group_tensor[0][line_indices][:, line_indices]
  for i1, i2 in zip(*np.where(affinity > _PARA_GROUP_THR)):
    line_grouping.union(i1, i2)

  line_groups = line_grouping.to_group()
  paragraphs = []
  for line_group in line_groups:
    paragraph = {'lines': []}
    for id_ in line_group:
      paragraph['lines'].append(lines[id_])
    if paragraph:
      paragraphs.append(paragraph)

  return paragraphs

class dummy_flag_class():
  def __init__(self, value) -> None:
    self.value = value
  
def initialize_flags(args_dict: dict[str, str]):
  global _GIN_FILE, _CKPT_PATH, _IMG_SIZE, _IMG_FILE, _IMG_DIR, _OUTPUT_PATH, _VIS_DIR, _VID
  _GIN_FILE = dummy_flag_class(args_dict['gin_file'])
  _CKPT_PATH = dummy_flag_class(args_dict['ckpt_path'])
  _IMG_SIZE = dummy_flag_class(args_dict.get('img_size', 1024))
  _IMG_FILE = dummy_flag_class(args_dict.get('img_file', []))
  _IMG_DIR = dummy_flag_class(args_dict.get('img_dir', []))
  _OUTPUT_PATH = dummy_flag_class(args_dict['output_path'])
  _VIS_DIR = dummy_flag_class(args_dict.get('vis_dir', None))
  _VID = dummy_flag_class(args_dict['vid'])

def detect_func(model, img_lists):
  vis_dis = _VIS_DIR.value
  vid = _VID.value
  output = {'annotations': []}
  line_contours = []

  for img_file in tqdm.tqdm(img_lists):
    if not vid:
      output['annotations'].append({
          'image_id': img_file.split('/')[-1].split('.')[0],
          'paragraphs': inference(img_file, model, False),
      })
    else:
      output['annotations'].append({
          'image_id': None,
          'paragraphs': inference(img_file, model, True),
      })

    key = output['annotations'][-1]['image_id']
    paragraphs = output['annotations'][-1]['paragraphs']

    if not vid:
      img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    else:
      img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)

    line_bnds = []
    para_ids = []                 # paragraph number of each line
    para_line_data: list[dict[str, list]] = []           # will store the four-point bouding boxes and other data of the lines within each para
    for idx, paragraph in enumerate(paragraphs):
      paragraph_points_list = []
      for line in paragraph['lines']:
        line_points_list = []
        for word in line['words']:
          line_contours.append(np.array(word['vertices'], np.int32).reshape((-1, 1, 2)))
          para_ids.append(idx)
          line_points_list.extend(word['vertices'])
        
        paragraph_points_list.extend(line_points_list)
        line_points = np.array(line_points_list, np.int32)  # (N,2)
        left = int(np.min(line_points[:, 0]))
        top = int(np.min(line_points[:, 1]))
        right = int(np.max(line_points[:, 0]))
        bottom = int(np.max(line_points[:, 1]))
        line_bnds.append(
            np.array([[[left, top]], [[right, top]], [[right, bottom]],
                      [[left, bottom]]], np.int32))
      
      para_points = np.array(paragraph_points_list, np.int32)  # (N,2)
      left = int(np.min(para_points[:, 0]))
      top = int(np.min(para_points[:, 1]))
      right = int(np.max(para_points[:, 0]))
      bottom = int(np.max(para_points[:, 1]))
      para_line_data.append({
        'bboxes': [], 
        'bbox_dims': [], 
        'line_ids': [], 
        'para_bbox':np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.int32)
        })

    i=0
    ocr_imgs = []
    bottom_left: list[tuple[float, float]] = []

    if not vid:
      word_cnt_img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    else:
      word_cnt_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)

    for idx, cnt in enumerate(line_contours):
      rect = cv2.minAreaRect(cnt)
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      box = np.where(box<0, 0, box) # take care of bounding box slightly out of the image
      # ADD CODE: add code to take care of bounding box crossing the right or bottom edges of the image
      w, h = rect[1]
      box_angle = rect[2]
      # the lines below are needed as opencv gives angles in a different way (see https://theailearner.com/tag/angle-of-rotation-by-cv2-minarearect/)
      '''if box_angle>45:
        box_angle = box_angle - 90
        w, h = h, w'''
      # Make sure that the image is lying on its longer side
      if w<=h:
        box_angle = box_angle - 90
        w, h = h, w

      dst_size = (np.int0(w), np.int0(h))
      if dst_size[0]==0 or dst_size[1]==0:
        continue
      
      rect_center = np.int0(np.array(rect[0]))  # center of bouding box
      
      para_line_data[para_ids[idx]]['bboxes'].append(box)
      para_line_data[para_ids[idx]]['bbox_dims'].append((w, h))
      '''
      i is used here rather than idx because if dst_size==0 the loop is skipped and we don't use the line.
      If idx was used, we wouldn't get continuous values 0, 1, 2, 3, ...
      '''
      para_line_data[para_ids[idx]]['line_ids'].append(i)
      '''
      Uncomment the line below to draw bounding boxes for the text detected
      '''
      #cv2.drawContours(word_cnt_img, [box], 0, (0,0,255), 2)
      '''
      Find the four extreme lines which cover the bouding boxes so we can crop them before we rotate and crop again.
      We do this so we don't have to rotate the entire image before cropping the bouding box.
      The names for the variables are switched in the y direction cause of opencv's coordinate system.
      For example the variable called top actually has the minimum y coordinate
      '''
      temp = np.max(box, axis=0)
      right = temp[0]
      bottom = temp[1]
      temp = np.min(box, axis=0)
      left = temp[0]
      top = temp[1]
      cropped_im = word_cnt_img[top: bottom, left: right]
      bottom_left.append((left, bottom))
      rect_center_cropped = rect_center - np.array([left, top])

      temp = np.int0(np.sqrt(np.int0(w)**2+np.int0(h)**2))
      rot_img = cropped_im
      top_border = np.int0((temp - cropped_im.shape[0])//2)
      side_border = np.int0((temp - cropped_im.shape[1])//2)
      if top_border<0:
        top_border = 0
      if side_border<0:
        side_border = 0
      # INSTEAD OF MAKING A BORDER JUST CROP TO A BIGGER IMAGE WHEN YOU INITIALLY CROP
      rot_img = cv2.copyMakeBorder(rot_img, top_border, top_border, side_border, side_border, cv2.BORDER_CONSTANT, value=0)
      rect_center_cropped -= np.array([-side_border, -top_border])
      rot_mat = cv2.getRotationMatrix2D(tuple(np.float64(rect_center_cropped[1::-1])), box_angle, 1.0)
      rot_img = cv2.warpAffine(rot_img, rot_mat, (temp, temp))
      rot_rect_center = np.matmul(rot_mat[:, 0:2], np.atleast_2d(rect_center_cropped).T).T[0]
      rot_rect_center = rot_rect_center + rot_mat[:, 2]
      #INSTEAD OF SHIFTING IT TO THE TOP-LEFT JUST CROP IT IN THE CENTER
      affine_disp = np.array([np.int0(w)//2, np.int0(h)//2]) - rot_rect_center
      translation_mat = np.float32([[1, 0, affine_disp[0]],
                                    [0, 1, affine_disp[1]]])
      rot_img = cv2.warpAffine(rot_img, translation_mat, dst_size)
      img_final = rot_img
      ocr_imgs.append(img_final)
      i = i+1
      # Uncomment the line below to visualize bouding boxes for the lines detected
      #cv2.drawContours(img, [box], 0, (0,0,255), 2)
  
  return ocr_imgs, img, para_line_data

# def MODIFIED_detect_func(model, img_lists):
#   vis_dis = _VIS_DIR.value
#   vid = _VID.value
#   output = {'annotations': []}
#   ocr_imgs = []
#   bottom_left = []
#   para_line_data = []           # will store the four-point bouding boxes and other data of the lines within each para
#   para_ids = []                 # paragraph number of each line

#   batch_paras = MODIFIED_inference(img_lists, model, vid)
#   full_imgs = []

#   for key, img_file in enumerate(img_lists):
#     paragraphs = batch_paras[key]

#     #key = output['annotations'][-1]['image_id']
#     #paragraphs = output['annotations'][-1]['paragraphs']

#     if not vid:
#       img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
#     else:
#       img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)

#     full_imgs.append(img)

#     word_contours = []
#     word_bnds = []
#     line_bnds = []
#     para_bnds = []
#     para_ids.append([])                 # paragraph number of each line
#     para_line_data.append([])           # will store the four-point bouding boxes and other data of the lines within each para
#     for idx, paragraph in enumerate(paragraphs):
#       para_line_data[-1].append({'bboxes': [], 'bbox_dims': [], 'line_ids': []})
#       paragraph_points_list = []
#       for line in paragraph['lines']:
#         line_points_list = []
#         for word in line['words']:
#           word_bnds.append(
#               np.array(word['vertices'], np.int32).reshape((-1, 1, 2)))
#           word_contours.append(np.array(word['vertices'], np.int32).reshape((-1, 1, 2))) # this is the same as words_bnds so use that
#           para_ids[-1].append(idx)
#           line_points_list.extend(word['vertices'])
        
#         paragraph_points_list.extend(line_points_list)
#         line_points = np.array(line_points_list, np.int32)  # (N,2)
#         left = int(np.min(line_points[:, 0]))
#         top = int(np.min(line_points[:, 1]))
#         right = int(np.max(line_points[:, 0]))
#         bottom = int(np.max(line_points[:, 1]))
#         line_bnds.append(
#             np.array([[[left, top]], [[right, top]], [[right, bottom]],
#                       [[left, bottom]]], np.int32))
      
#       para_points = np.array(paragraph_points_list, np.int32)  # (N,2)
#       left = int(np.min(para_points[:, 0]))
#       top = int(np.min(para_points[:, 1]))
#       right = int(np.max(para_points[:, 0]))
#       bottom = int(np.max(para_points[:, 1]))
#       para_bnds.append(
#           np.array([[[left, top]], [[right, top]], [[right, bottom]],
#                     [[left, bottom]]], np.int32))

#     i=0
#     ocr_imgs.append([])
#     bottom_left.append([])

#     if not vid:
#       word_cnt_img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
#     else:
#       word_cnt_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)

#     for idx, cnt in enumerate(word_contours):
#       rect = cv2.minAreaRect(cnt)
#       box = cv2.boxPoints(rect)
#       box = np.int0(box)
#       box = np.where(box<0, 0, box) # take care of bounding box slightly out of the image
#       # ADD CODE: add code to take care of bouding box crossing the right or bottom edges of the image
#       w, h = rect[1]
#       box_angle = rect[2]
#       # the lines below are needed as opencv gives angles in a different way (see https://theailearner.com/tag/angle-of-rotation-by-cv2-minarearect/)
#       if box_angle>45:
#         box_angle = box_angle - 90
#         w, h = h, w

#       dst_size = (np.int0(w), np.int0(h))
#       if dst_size[0]==0 or dst_size[1]==0:
#         continue
      
#       rect_center = np.int0(np.array(rect[0]))  # center of bouding box
      
#       para_line_data[-1][para_ids[-1][idx]]['bboxes'].append(box)
#       para_line_data[-1][para_ids[-1][idx]]['bbox_dims'].append((w, h))
#       '''
#       i is used here rather than idx because if dst_size==0 the loop is skipped and we don't use the line.
#       If idx was used, we wouldn't get continuous values 0, 1, 2, 3, ...
#       '''
#       para_line_data[-1][para_ids[-1][idx]]['line_ids'].append(i)
#       '''
#       Uncomment the line below to draw bounding boxes for the text detected
#       '''
#       #cv2.drawContours(word_cnt_img, [box], 0, (0,0,255), 2)
#       '''
#       Find the four extreme lines which cover the bouding boxes so we can crop them before we rotate and crop again.
#       We do this so we don't have to rotate the entire image before cropping the bouding box.
#       The names for the variables are switched in the y direction cause of opencv's coordinate system.
#       For example the variable called top actually has the minimum y coordinate
#       '''
#       temp = np.max(box, axis=0)
#       right = temp[0]
#       bottom = temp[1]
#       temp = np.min(box, axis=0)
#       left = temp[0]
#       top = temp[1]
#       cropped_im = word_cnt_img[top: bottom, left: right]
#       bottom_left[-1].append((left, bottom))
#       rect_center_cropped = rect_center - np.array([left, top])

#       temp = np.int0(np.sqrt(np.int0(w)**2+np.int0(h)**2))
#       rot_img = cropped_im
#       top_border = np.int0((temp - cropped_im.shape[0])//2)
#       side_border = np.int0((temp - cropped_im.shape[1])//2)
#       if top_border<0:
#         top_border = 0
#       if side_border<0:
#         side_border = 0
#       # INSTEAD OF MAKING A BORDER JUST CROP TO A BIGGER IMAGE WHEN YOU INITIALLY CROP
#       rot_img = cv2.copyMakeBorder(rot_img, top_border, top_border, side_border, side_border, cv2.BORDER_CONSTANT, value=0)
#       rect_center_cropped -= np.array([-side_border, -top_border])
#       rot_mat = cv2.getRotationMatrix2D(tuple(np.float64(rect_center_cropped[1::-1])), box_angle, 1.0)
#       rot_img = cv2.warpAffine(rot_img, rot_mat, (temp, temp))
#       rot_rect_center = np.matmul(rot_mat[:, 0:2], np.atleast_2d(rect_center_cropped).T).T[0]
#       rot_rect_center = rot_rect_center + rot_mat[:, 2]
#       #INSTEAD OF SHIFTING IT TO THE TOP-LEFT JUST CROP IT IN THE CENTER
#       affine_disp = np.array([np.int0(w)//2, np.int0(h)//2]) - rot_rect_center
#       translation_mat = np.float32([[1, 0, affine_disp[0]],
#                                     [0, 1, affine_disp[1]]])
#       rot_img = cv2.warpAffine(rot_img, translation_mat, dst_size)
#       img_final = rot_img

#       if not vid:
#         cv2.imwrite(os.path.join(vis_dis, f'{key}-{i}.jpg'), cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))

#       ocr_imgs[-1].append(img_final)
#       i = i+1
#       cv2.drawContours(img, [box], 0, (0,0,255), 2)

#     '''for name, bnds in zip(['paragraph', 'line', 'word'],
#                           [para_bnds, line_bnds, word_bnds]):
#       vis = cv2.polylines(img, bnds, True, (0, 0, 255), 2)
#       cv2.imwrite(os.path.join(vis_dis, f'{key}-{name}.jpg'),
#                   cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))'''

#   '''if not vid:
#     with tf.io.gfile.GFile(_OUTPUT_PATH.value, mode='w') as f:
#       f.write(json.dumps(output, ensure_ascii=False, indent=2))'''
  
#   return ocr_imgs, full_imgs, bottom_left, para_line_data, para_ids, para_bnds

def MODIFIED_inference(img_sources: Union[str, np.ndarray], model: tf.keras.layers.Layer, vid: bool) -> Dict[str, Any]:
  """Inference step."""
  imgs = []
  if not vid:
    for source in img_sources:
      imgs.append(cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB))
  else:
    for source in img_sources:
      imgs.append(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))

  img_ndarrays = []
  ratios = []
  for img in imgs:
    img_ndarray, ratio = MODIFIED__preprocess(img)
    img_ndarrays.append(img_ndarray)
    ratios.append(ratio)
  
  batched_input = np.asarray(img_ndarrays)
  output_dict = model.serve(batched_input)
  class_tensors = output_dict['classes'].numpy()
  mask_tensors = output_dict['masks'].numpy()
  group_tensors = output_dict['groups'].numpy()

  batched_paras = []

  for class_tensor, mask_tensor, group_tensor in zip(class_tensors, mask_tensors, group_tensors):
    class_tensor = np.expand_dims(class_tensor, axis=0)
    mask_tensor = np.expand_dims(mask_tensor, axis=0)
    group_tensor = np.expand_dims(group_tensor, axis=0)
    indices = np.where(class_tensor[0])[0].tolist()  # indices of positive slots.
    mask_list = [
        mask_tensor[0, :, :, index] for index in indices]  # List of mask ndarray.

    '''Uncomment line below to visualize line masks'''
    #visualize_masks(mask_list, indices)

    # Form lines and words
    lines = []
    line_indices = []
    for index, mask in tqdm.tqdm(zip(indices, mask_list)):
      line = {
          'words': [],
          'text': '',
      }

      contours, _ = cv2.findContours(
          (mask > 0.).astype(np.uint8),
          cv2.RETR_TREE,
          cv2.CHAIN_APPROX_SIMPLE)[-2:]

      '''
      Merging all the pseudo-word-level contours of a line into groups of line-level contour.
      If contours are very far apart they are kept in separate groups even though they are part of the same line.
      '''
      num_points = 0
      num_groups = 0
      cnt_groups = []
      for idx, contour in enumerate(contours):
        if (isinstance(contour, np.ndarray) and
            len(contour.shape) == 3 and
            contour.shape[0] > 2 and
            contour.shape[1] == 1 and
            contour.shape[2] == 2):
          num_points = num_points + contour.shape[0]
          for group in cnt_groups:
            for grouped_cnt in group:
              if is_same_group(contour, contours[grouped_cnt]):
                group.append(idx)
                break
            else:
              continue
            break
          else:
            cnt_groups.append([idx])
            num_groups = num_groups + 1

        else:
          logging.error('Invalid contour: %s, discarded', str(contour))
      
      merged_cnt = np.empty((num_groups, num_points, 1, 2), np.int32)
      for group_num, group in enumerate(cnt_groups):
        valid_idx = 0
        for cnt_idx in group:
          contour = contours[cnt_idx]
          merged_cnt[group_num, valid_idx: valid_idx+contour.shape[0]] = contour
          valid_idx = valid_idx + contour.shape[0]

        cnt_list = (merged_cnt[group_num, 0:valid_idx, 0] * ratio).astype(np.int32).tolist()
        line['words'].append({'text': '', 'vertices': cnt_list})
      '''End of merging all the pseudo-word-level contours of a line into one single line-level contour'''

      '''for contour in contours:
        if (isinstance(contour, np.ndarray) and
            len(contour.shape) == 3 and
            contour.shape[0] > 2 and
            contour.shape[1] == 1 and
            contour.shape[2] == 2):
          cnt_list = (contour[:, 0] * ratio).astype(np.int32).tolist()
          line['words'].append({'text': '', 'vertices': cnt_list})
        else:
          logging.error('Invalid contour: %s, discarded', str(contour))'''

      if line['words']:
        lines.append(line)
        line_indices.append(index)

    # Form paragraphs
    line_grouping = utilities.DisjointSet(len(line_indices))
    affinity = group_tensor[0][line_indices][:, line_indices]
    for i1, i2 in zip(*np.where(affinity > _PARA_GROUP_THR)):
      line_grouping.union(i1, i2)

    line_groups = line_grouping.to_group()
    paragraphs = []
    for line_group in line_groups:
      paragraph = {'lines': []}
      for id_ in line_group:
        paragraph['lines'].append(lines[id_])
      if paragraph:
        paragraphs.append(paragraph)
    
    batched_paras.append(paragraphs)

  return batched_paras

def MODIFIED__preprocess(raw_image: np.ndarray) -> Union[np.ndarray, float]:
  """Convert a raw image to properly resized, padded, and normalized ndarray."""
  # (1) convert to tf.Tensor and float32.
  img_tensor = tf.convert_to_tensor(raw_image, dtype=tf.float32)

  # (2) pad to square.
  height, width = img_tensor.shape[:2]
  maximum_side = tf.maximum(height, width)
  height_pad = maximum_side - height
  width_pad = maximum_side - width
  img_tensor = tf.pad(
      img_tensor, [[0, height_pad], [0, width_pad], [0, 0]],
      constant_values=127)
  ratio = maximum_side / _IMG_SIZE.value
  # (3) resize long side to the maximum length.
  img_tensor = tf.image.resize(
      img_tensor, (_IMG_SIZE.value, _IMG_SIZE.value))
  img_tensor = tf.cast(img_tensor, tf.uint8)

  # (4) normalize
  img_tensor = utilities.normalize_image_to_range(img_tensor)

  # (5) Add batch dimension and return as numpy array.
  return img_tensor.numpy(), float(ratio)

def is_same_group(ungrouped_cnt: np.ndarray, grouped_cnt: np.ndarray) -> bool:
  '''
  Determine whether two pseudo-word-level contours are part of the same line based on their proximity.
  This function assumes that contours are approximately rectangular.
  It may not work well for very irregularly shaped contours.
  Two contours are considered to be part of the same line if either the x or y distance is less than the x or y dimension of either contour.
  '''
  # If there was a simultaneous minmax function in numpy we could've used it :(
  max_ungrouped = np.max(ungrouped_cnt, axis=0)[0]
  max_grouped = np.max(grouped_cnt, axis=0)[0]
  min_ungrouped = np.min(ungrouped_cnt, axis=0)[0]
  min_grouped = np.min(grouped_cnt, axis=0)[0]
  dim_ungrouped = max_ungrouped - min_ungrouped
  dim_grouped = max_grouped - min_grouped
  min_dims = np.minimum(dim_ungrouped, dim_grouped)
  x_dist = abs(max(min_ungrouped[0], min_grouped[0]) - min(max_ungrouped[0], max_grouped[0]))
  y_dist = abs(max(min_ungrouped[1], min_grouped[1]) - min(max_ungrouped[1], max_grouped[1]))
  if x_dist>min_dims[0] or y_dist>min_dims[1]:
    return False
  return True


def visualize_masks(mask_list, indices):
  combined_mask = np.zeros(mask_list[0].shape).astype(int)
  combined_mask = np.uint8(np.repeat(combined_mask[:, :, np.newaxis], 3, 2))
  i = 0
  for cur_mask in mask_list:
    cur_mask = cur_mask.astype(int)
    cur_mask = np.where(cur_mask==1, 255, cur_mask)
    cur_mask = np.uint8(np.repeat(cur_mask[:, :, np.newaxis], 3, 2))
    cv2.imshow(f'masked image {indices[i]}', cur_mask)
    i = i+1
    combined_mask = combined_mask | cur_mask
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  combined_mask = np.where(combined_mask==1, 255, combined_mask)
  cv2.imshow('masked image', combined_mask)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
