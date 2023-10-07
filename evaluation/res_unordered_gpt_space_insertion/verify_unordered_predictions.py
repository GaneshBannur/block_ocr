import os, json, logging
from pathlib import Path

pred_path = os.path.join('./predictions.json')

if Path(pred_path).exists():
    with open(pred_path, 'r') as pred_fp:
        predictions = json.load(pred_fp)['predictions']

else:
    logging.error(f'Predictions not found at {pred_path}')

for img in predictions.values():
    for para in img:
        if para['para_text'] != ' '.join(para['line_texts']):
            logging.error(f"Para text: {para['para_text']}\nLine text: {para['line_texts']}")

print('Done!')
