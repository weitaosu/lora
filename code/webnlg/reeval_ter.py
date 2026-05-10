"""Add TER (Translation Edit Rate) to a finished run's metrics.json.

TER is the metric the LoRA paper reports for WebNLG (lower is better).
We compute it post-hoc from predictions.jsonl so no retraining is needed.

Usage:
    python reeval_ter.py ../../results/webnlg/lora_webnlg_v2.1
    python reeval_ter.py ../../results/webnlg/full_ft_v2.1
"""
import json
import sys
from pathlib import Path
import sacrebleu


def compute_ter(predictions: list[str], references: list[list[str]]) -> float:
    """Corpus-level TER. Pads ragged refs to a rectangular structure for sacrebleu."""
    max_refs = max(len(r) for r in references)
    refs_rect = [[r[i] if i < len(r) else r[0] for r in references] for i in range(max_refs)]
    ter = sacrebleu.corpus_ter(predictions, refs_rect)
    return float(ter.score)   # sacrebleu returns 0-100 scale


def main(run_dir: Path) -> None:
    preds_path = run_dir / 'predictions.jsonl'
    metrics_path = run_dir / 'metrics.json'
    if not preds_path.exists():
        raise FileNotFoundError(preds_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    predictions, references = [], []
    with open(preds_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            # use a single space placeholder for empty preds so sacrebleu doesn't crash
            predictions.append(r['pred'] if r['pred'].strip() else ' ')
            references.append(r['refs'])
    print(f'Scoring {len(predictions)} predictions in {run_dir.name}')

    ter_pct = compute_ter(predictions, references)   # 0-100
    ter_unit = ter_pct / 100.0                        # paper's 0-1 scale

    m = json.load(open(metrics_path))
    m['TER']      = ter_pct          # 0-100 (sacrebleu native)
    m['TER_unit'] = ter_unit         # 0-1 (paper's scale, lower is better)
    json.dump(m, open(metrics_path, 'w'), indent=2)
    print(f'  TER: {ter_pct:.4f} (0-100 scale) = {ter_unit:.4f} (paper 0-1 scale, lower is better)')
    print(f'  Updated {metrics_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
