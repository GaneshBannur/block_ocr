import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from collections import defaultdict

def load_pretrained_model_and_transforms():
    '''
    Load model and image transforms
    '''
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform

def load_model_and_transforms():
    '''
    Load model and image transforms
    '''
    parseq = load_from_checkpoint('parseq_with_space.ckpt').eval()
    parseq = parseq.to("cpu")
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform

@torch.no_grad()
def recognize_text(parseq, img_transform, imgs) -> tuple[list[str], list[torch.Tensor]]:
    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    temp = []
    indices = defaultdict(lambda: {})
    for idx, x in enumerate(imgs):
        cur = 0
        indices[idx]['start'] = len(temp)
        while cur<x.shape[1]:
            temp.append(x[:,cur:cur+x.shape[0]*4, :])
            cur = cur+x.shape[0]*4
        indices[idx]['end'] = len(temp)
    imgs = temp
    imgs = [img_transform(Image.fromarray(x)).unsqueeze(0) for x in imgs]
    imgs = torch.vstack(imgs)

    logits = parseq(imgs)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    labels, confidences = parseq.tokenizer.decode(pred)
    temp_labels = []
    temp_confidences = []
    for img_index in indices.values():
        temp_labels.append(''.join(labels[img_index['start']: img_index['end']]))
        # Every partial image will produce an [EOS]. We don't need the confidences for these intermediate [EOS] tokens
        full_confidence = torch.cat(
            tuple(
                [partial_confidence[:-1] for partial_confidence in confidences[img_index['start']: img_index['end']]]
                )
            )
        # However we keep the last [EOS] token's confidence
        full_confidence = torch.cat((full_confidence, confidences[img_index['start']: img_index['end']][-1][-1:]))
        temp_confidences.append(full_confidence)
    labels = temp_labels
    confidences = temp_confidences
    return labels, confidences

@torch.no_grad()
def BATCHED_recognize_text(parseq, img_transform, imgs) -> tuple[list[str], list[torch.Tensor]]:
    #img = Image.open('/Users/snoopy/Desktop/Other/IISc/test_imgs/recognition/blurry.png').convert('RGB')
    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    input_shapes = [0]
    for single_batch in imgs:
        input_shapes.append(len(single_batch))
    for i in range(1, len(input_shapes)):
        input_shapes[i] = input_shapes[i] + input_shapes[i-1]
    
    transformed = []
    for single_batch in imgs:
        transformed.append([img_transform(Image.fromarray(x)).unsqueeze(0) for x in single_batch])
    imgs = transformed
    stacked = []
    for single_batch in imgs:
        stacked.append(torch.vstack(single_batch))
    imgs = stacked
    #imgs = [img_transform(x).unsqueeze(0) for x in imgs]
    imgs = torch.vstack(imgs)
    logits = parseq(imgs)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    labels, confidences = parseq.tokenizer.decode(pred)
    labels = [labels[input_shapes[i]: input_shapes[i+1]] for i in range(len(input_shapes)-1)]
    confidences = [confidences[input_shapes[i]: input_shapes[i+1]] for i in range(len(input_shapes)-1)]
    return labels, confidences