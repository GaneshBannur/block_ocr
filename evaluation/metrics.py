from evaluate import load
from meteor import Meteor
from jiwer import wer as word_error_rate
from sacrebleu import TER
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from bert_score import BERTScorer
from bleurt import score as bleurt_score
from Levenshtein import distance as ld
from Levenshtein import hamming as hd
from Levenshtein import jaro_winkler as jw
from fastDamerauLevenshtein import damerauLevenshtein
from difflib import SequenceMatcher

checkpoint = "bleurt/BLEURT-20"
BLEURT_scorer = bleurt_score.BleurtScorer(checkpoint)
BERTScore_scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')
sentence_transformer_model  = SentenceTransformer('all-mpnet-base-v2')
google_bleu = load("google_bleu")
google_rouge = load('rouge')
huggingface_meteor = Meteor()
huggingface_meteor._download_and_prepare(None)
sacreblue_ter = TER()
gestalt_matcher = SequenceMatcher()

def bleu(gt_string, pred_string):
    return google_bleu.compute(predictions=[pred_string], references=[[gt_string]])['google_bleu']

def rouge(gt_string, pred_string):
    return google_rouge.compute(predictions=[pred_string], references=[gt_string])

def meteor(gt_string, pred_string):
    return huggingface_meteor._compute(predictions=[pred_string], references=[gt_string])['meteor']

def wer(gt_string, pred_string):
    return word_error_rate(hypothesis=pred_string, reference=gt_string)

def ter(gt_string, pred_string):
    return sacreblue_ter.sentence_score(hypothesis=pred_string, references=[gt_string]).score / 100

def sent_transformer(gt_string, pred_string):
    return float(cos_sim(
                    sentence_transformer_model.encode(pred_string), 
                    sentence_transformer_model.encode(gt_string)
                    ).numpy()[0][0])

def BERTScore(gt_string, pred_string):
    return float(BERTScore_scorer.score([pred_string], [gt_string])[2].numpy()[0])

def BLEURT(gt_string, pred_string):
    return BLEURT_scorer.score(candidates=[pred_string], references=[gt_string])[0]

def levenshtein(gt_string, pred_string):
    return ld(pred_string, gt_string)

def hamming(gt_string, pred_string):
    return hd(pred_string, gt_string)

def jaro_winkler_similarity(gt_string, pred_string):
    return jw(pred_string, gt_string)

def damerau_levenshtein(gt_string, pred_string):
    return damerauLevenshtein(pred_string, gt_string, similarity=False)

def gestalt_similarity(gt_string, pred_string):
    gestalt_matcher.set_seqs(pred_string, gt_string)
    return gestalt_matcher.ratio()

def compute_all_metrics(gt_string, pred_string):
    norm_len = max(len(gt_string), len(pred_string))
    rouge_scores = rouge(gt_string, pred_string)
    levenshtein_score = levenshtein(gt_string, pred_string)
    hamming_score = hamming(gt_string, pred_string)
    damerau_levenshtein_score = damerau_levenshtein(gt_string, pred_string)
    return {
        'bleu': bleu(gt_string, pred_string),
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'rougeLsum': rouge_scores['rougeLsum'],
        'meteor': meteor(gt_string, pred_string),
        'wer': wer(gt_string, pred_string),
        'ter': ter(gt_string, pred_string),
        'sent_transformer': sent_transformer(gt_string, pred_string),
        'bert_score': BERTScore(gt_string, pred_string),
        'bleurt': BLEURT(gt_string, pred_string),
        'levenshtein': levenshtein_score,
        'levenshtein_norm': levenshtein_score/norm_len,
        'hamming': hamming_score,
        'hamming_norm': hamming_score/norm_len,
        'jaro_winkler_similarity': jaro_winkler_similarity(gt_string, pred_string),
        'damerau_levenshtein': damerau_levenshtein_score,
        'damerau_levenshtein_norm': damerau_levenshtein_score/norm_len,
        'gestalt_similarity': gestalt_similarity(gt_string, pred_string)
    }
