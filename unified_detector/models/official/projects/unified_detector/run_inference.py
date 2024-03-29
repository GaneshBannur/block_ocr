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

import json
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

'''
# importing modules for ocr
import importlib
ocr_utils = importlib.import_module('semimtr-text-recognition.semimtr.utils.utils')
Config = ocr_utils.Config
CharsetMapper = ocr_utils.CharsetMapper
ocr_demo = importlib.import_module('semimtr-text-recognition.demo')
get_model = ocr_demo.get_model
load = ocr_demo.load
get_ocr_prediction = ocr_demo.get_ocr_prediction
'''
'''
# importing modules for ocr
ocr_path = '/Users/snoopy/Desktop/Other/IISc/text-recognition/semimtr/semimtr-my-version/semimtr-text-recognition/'
import importlib.util, sys
ocr_utils_file_path = ocr_path+'semimtr/utils/utils.py'
ocr_utils_module_name = 'semimtr.utils.utils'
ocr_utils_spec = importlib.util.spec_from_file_location(ocr_utils_module_name, ocr_utils_file_path)
ocr_utils_module = importlib.util.module_from_spec(ocr_utils_spec)
sys.modules[ocr_utils_module_name] = ocr_utils_module
ocr_utils_spec.loader.exec_module(ocr_utils_module)
from semimtr.utils.utils import Config, Logger, CharsetMapper

ocr_demo_file_path = ocr_path+'demo.py'
ocr_demo_module_name = 'demo'
ocr_demo_spec = importlib.util.spec_from_file_location(ocr_demo_module_name, ocr_demo_file_path)
ocr_demo_module = importlib.util.module_from_spec(ocr_demo_spec)
sys.modules[ocr_demo_module_name] = ocr_demo_module
ocr_demo_spec.loader.exec_module(ocr_demo_module)
from demo import get_model, load, get_ocr_prediction
'''


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


def inference(img_file: str, model: tf.keras.layers.Layer) -> Dict[str, Any]:
  """Inference step."""
  img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
  img_ndarray, ratio = _preprocess(img)

  output_dict = model.serve(img_ndarray)
  class_tensor = output_dict['classes'].numpy()
  mask_tensor = output_dict['masks'].numpy()
  group_tensor = output_dict['groups'].numpy()

  indices = np.where(class_tensor[0])[0].tolist()  # indices of positive slots.
  mask_list = [
      mask_tensor[0, :, :, index] for index in indices]  # List of mask ndarray.

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
    for contour in contours:
      if (isinstance(contour, np.ndarray) and
          len(contour.shape) == 3 and
          contour.shape[0] > 2 and
          contour.shape[1] == 1 and
          contour.shape[2] == 2):
        cnt_list = (contour[:, 0] * ratio).astype(np.int32).tolist()
        line['words'].append({'text': '', 'vertices': cnt_list})
      else:
        logging.error('Invalid contour: %s, discarded', str(contour))
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

def ocr_init():
  '''
  ocr_args = {'config': ocr_path+'configs/semimtr_finetune.yaml',
              'cuda': -1,
              'checkpoint': ocr_path+'workdir/semimtr_model.pth',
              'model_eval': 'alignment'}
  '''
  ocr_args = {'config': 'configs/semimtr_finetune.yaml',
              'cuda': -1,
              'checkpoint': 'workdir/semimtr_model.pth',
              'model_eval': 'alignment'}

  ocr_config = Config(ocr_args['config'])
  ocr_config.model_checkpoint = ocr_args['checkpoint']
  ocr_config.model_eval = ocr_args['model_eval']
  ocr_config.global_phase = 'test'
  ocr_config.model_vision_checkpoint, ocr_config.model_language_checkpoint = None, None
  ocr_device = 'cpu' if ocr_args['cuda'] < 0 else f'cuda:{ocr_args["cuda"]}'
  ocr_model = get_model(ocr_config).to(ocr_device)
  ocr_model = load(ocr_model, ocr_config.model_checkpoint, device=ocr_device)
  charset = CharsetMapper(filename=ocr_config.dataset_charset_path,
                          max_length=ocr_config.dataset_max_length + 1)
  return ocr_config, ocr_device, ocr_model, charset

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Get list of images
  img_lists = []
  img_lists.extend(_IMG_FILE.value)
  for img_dir in _IMG_DIR.value:
    img_lists.extend(tf.io.gfile.glob(os.path.join(img_dir, '*')))

  logging.info('Total number of input images: %d', len(img_lists))

  model = load_model()

  vis_dis = _VIS_DIR.value

  output = {'annotations': []}

  # my additions

  #ocr_config, ocr_device, ocr_model, charset = ocr_init()
  word_contours = []

  for img_file in tqdm.tqdm(img_lists):
    output['annotations'].append({
        'image_id': img_file.split('/')[-1].split('.')[0],
        'paragraphs': inference(img_file, model),
    })

    if vis_dis:
      key = output['annotations'][-1]['image_id']
      paragraphs = output['annotations'][-1]['paragraphs']
      img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
      word_bnds = []
      line_bnds = []
      para_bnds = []
      for paragraph in paragraphs:
        paragraph_points_list = []
        for line in paragraph['lines']:
          line_points_list = []
          for word in line['words']:
            word_bnds.append(
                np.array(word['vertices'], np.int32).reshape((-1, 1, 2)))

            # my additions
            word_contours.append(np.array(word['vertices'], np.int32).reshape((-1, 1, 2))) #this is the same as words_bnds so use that

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
        para_bnds.append(
            np.array([[[left, top]], [[right, top]], [[right, bottom]],
                      [[left, bottom]]], np.int32))

      # my additions
      i=0 # DEBUG: REMOVE THIS LATER
      word_cnt_img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
      for cnt in word_contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = np.where(box<0, 0, box) # take care of bounding box slightly out of the image
        '''
        Uncomment the line below to draw bounding boxes for the text detected
        '''
        #cv2.drawContours(word_cnt_img, [box], 0, (0,0,255), 2)
        top_left_cnt = np.full(box.shape, np.array(cnt[0]))             # the first value in the word prediction is the top left point
        top_left = np.argmin(np.square(top_left_cnt - box).sum(axis=1)) # find the point in the bounding box closest to the top left point  
        rect_center = np.int0(np.array(rect[0]))
        coordinate_diff = box-rect_center                                               # we need angles with respect to the center so we transform the coordinates to have the center as the origin
        angles = np.arctan(coordinate_diff[:, 1]/coordinate_diff[:, 0]) * (180.0/np.pi) # calculate slope and take arctan to get angle made by point-center line

        '''
        np.arctan returns angles between -90 and 90 so we need to add 180 or 360 depending on the quadrant of the
        point to get positive angles
        '''
        for row in range(coordinate_diff.shape[0]):
          point = coordinate_diff[row]
          if point[0]<0 and point[1]>=0 or point[0]<=0 and point[1]<0:
            angles[row] += 180
          elif point[0]>0 and point[1]<0:
            angles[row] += 360
                
        order = np.argsort(angles)
        top_left_index = np.asarray(order==top_left).nonzero()[0][0]  # find the index of top left element and rotate array to bring it to the front
        order = np.roll(order, -top_left_index)
        
        # plots the four vertices of the bouding box with colours red, green, blue, black, in order of orientation
        '''
        cv2.circle(word_cnt_img, box[order[0]], radius=10, color = (255, 0, 0), thickness=-1)
        cv2.circle(word_cnt_img, box[order[1]], radius=10, color = (0, 255, 0), thickness=-1)
        cv2.circle(word_cnt_img, box[order[2]], radius=10, color = (0, 0, 255), thickness=-1)
        cv2.circle(word_cnt_img, box[order[3]], radius=10, color = (0, 0, 0), thickness=-1)
        '''

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
        
        '''box_cropped = box - np.array([left, top])
        rect_center_cropped = rect_center - np.array([left, top])
        midpoint_2_3 = (box_cropped[order[1]]+box_cropped[order[2]])/2.0  # USING THE 1,2,3,4 ORDERING OF POINTS IN VARIABLE NAMING HERE. CHANGE THIS OR EVERYTHING ELSE
        coordinate_diff = midpoint_2_3 - rect_center_cropped                          # we need angles with respect to the center so we transform the coordinates
        rot_angle = np.arctan(coordinate_diff[1]/coordinate_diff[0]) * (180.0/np.pi)  # calculate slope and take arctan to get angle made by point-center line
        ''''''
        np.arctan returns angles between -pi/2 and pi/2 so we need to add 180 or 360 depending on the quadrant of the
        point to get positive angles
        ''''''
        if coordinate_diff[0]<0 and coordinate_diff[1]>=0 or coordinate_diff[0]<=0 and coordinate_diff[1]<0:
          rot_angle += 180
        elif coordinate_diff[0]>0 and coordinate_diff[1]<0:
          rot_angle += 360'''
        
        rect_center_cropped = rect_center - np.array([left, top])
        w, h = rect[1]
        h_disp, w_disp = np.int0(h)//2, np.int0(w)//2
        top_f = rect_center_cropped[1]-h_disp
        bottom_f = rect_center_cropped[1]+h_disp
        left_f = rect_center_cropped[0]-w_disp
        right_f = rect_center_cropped[0]+w_disp
        # if the model got the orientation wrong we don't try to rotate the image
        if top_f<0 or left_f<0:
          cv2.imwrite(os.path.join(vis_dis, f'{key}-{i}.jpg'), cv2.cvtColor(cropped_im, cv2.COLOR_RGB2BGR))
          ocr_im = cropped_im
        else:
          angle_2_3 = np.abs(angles[order[1]] - angles[order[2]])
          if angle_2_3 > 180:
            angle_2_3 = 360 - angle_2_3
          midpoint_2_3_angle = np.remainder(angles[order[1]] + angle_2_3/2.0, 360)
          midpoint_2_3_angle = 360 - midpoint_2_3_angle
          rot_mat = cv2.getRotationMatrix2D(tuple(np.float64(rect_center_cropped[1::-1])), -midpoint_2_3_angle, 1.0)
          dst_approx = np.sum(np.array(cropped_im.shape)[0:2]) * 2   # must be an integer
          dst_size = (dst_approx, dst_approx)                                     # max possible height and width after rotation are the diagonals
          crop_im_center = np.array(cropped_im.shape)[0:2]//2
          affine_disp = crop_im_center - rect_center_cropped
          # calculate the displacement between the centers of the bounding box and image and move the bounding box to the center
          rot_mat[0, 2] -= affine_disp[0] * 0.5 # I don't know why - works here instead of +
          rot_mat[1, 2] += affine_disp[1] * 0.5  # DEBUG: DECIDE THIS FACTOR
          rot_img = cv2.warpAffine(cropped_im, rot_mat, dst_size)

          img_final = rot_img[top_f: bottom_f, left_f: right_f]
          cv2.imwrite(os.path.join(vis_dis, f'{key}-{i}.jpg'), cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))
          ocr_im = img_final
        
        # DEBUG: CONSIDERING ONLY CROPPED IMAGES, NOT ROTATED
        #cv2.imwrite(os.path.join(vis_dis, f'{key}-{i}.jpg'), cv2.cvtColor(cropped_im, cv2.COLOR_RGB2BGR))

        i = i+1
        '''
        ocr_prediction = get_ocr_prediction(ocr_im, ocr_config, ocr_device, ocr_model, charset)
        for ocr_text in ocr_prediction:
          cv2.putText(word_cnt_img, ocr_text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        '''
      
      cv2.imwrite(os.path.join(vis_dis, f'{key}-word-contours.jpg'), cv2.cvtColor(word_cnt_img, cv2.COLOR_RGB2BGR))

      for name, bnds in zip(['paragraph', 'line', 'word'],
                            [para_bnds, line_bnds, word_bnds]):
        vis = cv2.polylines(img, bnds, True, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(vis_dis, f'{key}-{name}.jpg'),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

  with tf.io.gfile.GFile(_OUTPUT_PATH.value, mode='w') as f:
    f.write(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  flags.mark_flags_as_required(['gin_file', 'ckpt_path', 'output_path'])
  app.run(main)
