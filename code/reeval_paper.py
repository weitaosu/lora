"""Re-score a finished run using the paper's metric setup, with Seen/Unseen/All split.

Paper (Hu et al. 2021, Table 14, GPT-2 M, WebNLG):
- Splits test set by category into:
    Seen   (10 cats): Airport, Astronaut, Building, City, ComicsCharacter,
                      Food, Monument, SportsTeam, University, WrittenWork
    Unseen (5 cats):  Artist, Athlete, CelestialBody, MeanOfTransportation, Politician
- Reports per split: BLEU, METEOR (Java implementation), TER
- METEOR uses the official Java jar (bundled with pycocoevalcap), NOT NLTK.

Usage:
    python reeval_paper.py ../results/lora_webnlg_v2.1
    python reeval_paper.py ../results/full_ft_v2.1

Requires Java (we install OpenJDK 17 via `pip install install-jdk` if needed).
"""
import json
import os
import sys
from pathlib import Path

# Point pycocoevalcap at our locally-installed JDK before importing it.
JDK_DIR = Path(os.path.expanduser('~/.jdks'))
if JDK_DIR.exists():
    candidates = sorted(JDK_DIR.glob('jdk-*'))
    if candidates:
        java_home = candidates[-1]   # newest
        os.environ['JAVA_HOME'] = str(java_home)
        os.environ['PATH'] = str(java_home / 'bin') + os.pathsep + os.environ.get('PATH', '')

import sacrebleu
from pycocoevalcap.meteor.meteor import Meteor

sys.path.insert(0, str(Path(__file__).parent))
from webnlg_loader import load_webnlg

SEEN = {
    'Airport', 'Astronaut', 'Building', 'City', 'ComicsCharacter',
    'Food', 'Monument', 'SportsTeam', 'University', 'WrittenWork',
}
UNSEEN = {'Artist', 'Athlete', 'CelestialBody', 'MeanOfTransportation', 'Politician'}


def split_metrics(predictions: list[str], references: list[list[str]]) -> dict:
    """BLEU + Java METEOR + TER on a list of predictions/references."""
    if not predictions:
        return {'BLEU': float('nan'), 'METEOR': float('nan'), 'TER': float('nan'), 'n': 0}

    # Pad refs to a rectangular structure for sacrebleu.
    max_refs = max(len(r) for r in references)
    refs_rect = [[r[i] if i < len(r) else r[0] for r in references] for i in range(max_refs)]

    safe_preds = [p if p.strip() else ' ' for p in predictions]
    bleu = float(sacrebleu.corpus_bleu(safe_preds, refs_rect).score)
    ter  = float(sacrebleu.corpus_ter(safe_preds, refs_rect).score)

    # Java METEOR via pycocoevalcap. Returns score on 0-1 scale.
    gts = {i: list(refs) for i, refs in enumerate(references)}
    res = {i: [predictions[i] if predictions[i].strip() else ' '] for i in range(len(predictions))}
    meteor, _ = Meteor().compute_score(gts, res)

    return {
        'BLEU':   bleu,           # 0-100
        'METEOR': float(meteor),  # 0-1 (paper scale)
        'TER':    ter / 100.0,    # 0-1 (paper scale, lower is better)
        'n':      len(predictions),
    }


def main(run_dir: Path) -> None:
    preds_path   = run_dir / 'predictions.jsonl'
    metrics_path = run_dir / 'metrics.json'

    # Load predictions and re-derive category from the (cached) WebNLG test set
    # so we can split into Seen/Unseen/All exactly like the paper.
    _, _, test_rows = load_webnlg(Path(__file__).parent.parent / 'data' / 'raw',
                                  version='v2.1')
    src_to_cat = {r['src']: r.get('category', '') for r in test_rows}

    splits = {'Seen': [], 'Unseen': [], 'All': []}
    with open(preds_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            cat = src_to_cat.get(r['src'], '')
            row = (r['pred'], r['refs'], cat)
            splits['All'].append(row)
            if cat in SEEN:
                splits['Seen'].append(row)
            elif cat in UNSEEN:
                splits['Unseen'].append(row)

    print(f'Run: {run_dir.name}')
    print(f'  All n={len(splits["All"])}  Seen n={len(splits["Seen"])}  Unseen n={len(splits["Unseen"])}')

    paper_split = {}
    for name, rows in splits.items():
        preds = [r[0] for r in rows]
        refs  = [r[1] for r in rows]
        m = split_metrics(preds, refs)
        paper_split[name] = m
        print(f'  {name:6s}  BLEU={m["BLEU"]:6.2f}  METEOR={m["METEOR"]:.4f}  TER={m["TER"]:.4f}  (n={m["n"]})')

    # Append paper-style metrics to metrics.json without disturbing existing keys.
    existing = json.load(open(metrics_path))
    existing['paper_metrics'] = {
        'note': 'BLEU + Java METEOR (pycocoevalcap, official meteor-1.5.jar) + TER, split by 2017 challenge categories',
        'seen_categories':   sorted(SEEN),
        'unseen_categories': sorted(UNSEEN),
        'splits': paper_split,
    }
    json.dump(existing, open(metrics_path, 'w'), indent=2)
    print(f'  Updated {metrics_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(Path(sys.argv[1]).resolve())
