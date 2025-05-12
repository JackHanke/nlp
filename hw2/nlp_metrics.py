from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def compute_bleu(reference: list[str], candidate: list[str]) -> float:
    """
    Compute BLEU score for a single sentence.
    """
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], candidate, smoothing_function=smoothie)


def compute_rouge(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)


def compute_bertscore(references: list[str], candidates: list[str], lang: str = "en") -> tuple[list[float], list[float], list[float]]:
    """
    Compute BERTScore precision, recall, and F1.
    """
    P, R, F1 = bert_score_fn(candidates, references, lang=lang, verbose=False)
    return P.tolist(), R.tolist(), F1.tolist()