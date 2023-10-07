import os, json, logging, shutil
from pathlib import Path
from datetime import datetime

pred_path = os.path.join('./predictions.json')
temp_path = os.path.join('./temp.json')

if Path(pred_path).exists():
    with open(pred_path, 'r') as pred_fp:
        predictions = json.load(pred_fp)['predictions']

else:
    logging.error(f'Predictions not found at {pred_path}')

def save_preds(preds):
    cur_datetime = str(datetime.now())
    try:
        Path(os.path.join(
            '.'
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

for img in predictions.values():
    for para in img:
        para['para_text'] = ' '.join(para['line_texts'])

save_preds(predictions)