

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.nist_score import corpus_nist
from rouge import Rouge
from nltk.metrics.distance import edit_distance

def calculate_metrics(ref, hyp):
    smoothing = SmoothingFunction().method4
    try:
        bleu = corpus_bleu(ref, hyp, smoothing_function=smoothing)
        nist = corpus_nist(ref, hyp, n=4)
    except:
        bleu = 0
        nist = 0

    # Calculate dist
    total_len = 0.0
    edi = 0.0
    for r, h in zip(ref, hyp):
            pred = ' '.join([str(x) for x in h])  
            target = ' '.join([str(x) for x in r[0]])        
            total_len += max(len(pred), len(target))
            edi += edit_distance(pred, target)


    dist = round(1-edi/total_len,3)

    rouge = Rouge()
    hyp_joined = [' '.join([str(x) for x in h]) for h in hyp]
    ref_joined = [' '.join([str(x) for x in r[0]]) for r in ref]
    try:
        scores = rouge.get_scores(hyp_joined, ref_joined, avg=True)
        return bleu, nist, dist, scores['rouge-2']['f'], scores['rouge-l']['f']
    except:
        return bleu, nist, dist, 0.0, 0.0
