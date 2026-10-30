"""Recompute compact real-data evidence with only the Python standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'sensitivity_hgg',
           'specificity_lgg', 'roc_auc')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, context):
    require(math.isclose(float(actual), float(expected), abs_tol=1e-10, rel_tol=1e-10),
            f'{context}: recomputed {actual}, saved {expected}')


def calculate(rows):
    require(bool(rows), 'Empty prediction file')
    ids = [r['patient_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'Duplicate prediction patient IDs')
    cm = [[0, 0], [0, 0]]
    pairs = []
    for row in rows:
        label, score = int(row['label']), float(row['probability_hgg'])
        require(label in (0, 1) and math.isfinite(score) and 0 <= score <= 1,
                f'Invalid prediction for {row["patient_id"]}')
        pred = int(score >= 0.5)
        if 'prediction' in row:
            require(int(row['prediction']) == pred, 'Saved prediction differs from threshold 0.5')
        if 'label_name' in row:
            require(row['label_name'] == ('LGG', 'HGG')[label], 'Grade label mismatch')
        if 'prediction_name' in row:
            require(row['prediction_name'] == ('LGG', 'HGG')[pred], 'Prediction name mismatch')
        if 'correct' in row:
            require(row['correct'].lower() == str(label == pred).lower(), 'Correctness flag mismatch')
        cm[label][pred] += 1
        pairs.append((score, label))
    tn, fp = cm[0]
    fn, tp = cm[1]
    require(tn + fp > 0 and tp + fn > 0, 'Both classes are required')
    per_class = {}
    for i, name in enumerate(('LGG', 'HGG')):
        precision = cm[i][i] / sum(cm[j][i] for j in (0, 1)) if sum(cm[j][i] for j in (0, 1)) else 0
        recall = cm[i][i] / sum(cm[i])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': sum(cm[i])}
    # Mann-Whitney rank formulation of ROC-AUC, with average ranks for ties.
    pairs.sort()
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(label for _, label in pairs[i:j])
        i = j
    positive, negative = tp + fn, tn + fp
    auc = (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    return {'n_patients': len(rows), 'accuracy': (tn + tp) / len(rows),
            'balanced_accuracy': (tn / negative + tp / positive) / 2,
            'macro_f1': sum(v['f1'] for v in per_class.values()) / 2,
            'sensitivity_hgg': tp / positive, 'specificity_lgg': tn / negative,
            'roc_auc': auc, 'confusion_matrix_lgg_hgg': cm, 'per_class': per_class}


def check_metrics(rows, saved, context):
    actual = calculate(rows)
    require(actual['n_patients'] == saved['n_patients'], f'{context}: patient count')
    require(actual['confusion_matrix_lgg_hgg'] == saved['confusion_matrix_lgg_hgg'], f'{context}: confusion matrix')
    for key in METRICS:
        same(actual[key], saved[key], f'{context}/{key}')
    for name in ('LGG', 'HGG'):
        for key in ('precision', 'recall', 'f1', 'support'):
            same(actual['per_class'][name][key], saved['per_class'][name][key], f'{context}/{name}/{key}')
    return actual


def split_ids(manifest):
    ids = [r['patient_id'] for r in manifest]
    require(len(ids) == len(set(ids)) == 369, 'Manifest must contain 369 unique patients')
    parts = {k: {r['patient_id'] for r in manifest if r['split'] == k} for k in ('train', 'val', 'test')}
    require(set.union(*parts.values()) == set(ids), 'Unassigned patient split')
    require(not (parts['train'] & parts['val'] or parts['train'] & parts['test'] or parts['val'] & parts['test']),
            'Patient overlap between roles')
    return parts


def check_run(folder, manifest):
    saved = read_json(folder / 'test_metrics.json')
    predictions = read_csv(folder / 'test_predictions.csv')
    parts = split_ids(manifest)
    require({r['patient_id'] for r in predictions} == parts['test'], f'{folder.name}: test membership')
    labels = {r['patient_id']: int(r['label']) for r in manifest}
    require(all(int(r['label']) == labels[r['patient_id']] for r in predictions), 'Prediction/manifest label mismatch')
    check_metrics(predictions, saved, folder.name)
    for role in ('train', 'val', 'test'):
        counts = Counter(r['label_name'] for r in manifest if r['split'] == role)
        require(dict(counts) == saved['split_counts'][role], f'{folder.name}: split class counts')
    history = read_csv(folder / 'history.csv')
    best = max(history, key=lambda r: float(r['val_balanced_accuracy']))
    require(int(best['epoch']) == saved['best_epoch'], f'{folder.name}: selected epoch')
    same(best['val_balanced_accuracy'], saved['best_validation_balanced_accuracy'], f'{folder.name}: validation selection')
    require([int(r['epoch']) for r in history] == list(range(1, len(history) + 1)), 'Non-sequential history')
    return saved, predictions


def verify(root=ROOT):
    root = Path(root)
    evidence = root / 'results/gpu'
    manifest = read_csv(evidence / 'manifest.csv')
    parts = split_ids(manifest)
    require({r['label_name'] for r in manifest} == {'LGG', 'HGG'}, 'Unexpected grade labels')
    require(Counter(r['label_name'] for r in manifest) == {'LGG': 76, 'HGG': 293}, 'Cohort counts')
    require({k: len(v) for k, v in parts.items()} == {'train': 257, 'val': 56, 'test': 56}, 'Fixed split counts')
    audit = read_csv(evidence / 'data_audit.csv')
    require(len(audit) == 369 and all(r['valid'].lower() == 'true' for r in audit), 'Data audit did not pass')
    fixed, _ = check_run(evidence / 'fixed_split', manifest)
    brain, _ = check_run(evidence / 'whole_brain_fixed', manifest)
    cv = read_json(evidence / 'cv5/cross_validation_metrics.json')
    oof = read_csv(evidence / 'cv5/out_of_fold_predictions.csv')
    check_metrics(oof, cv, 'pooled OOF')
    oof_map = {r['patient_id']: r for r in oof}
    require(set(oof_map) == {r['patient_id'] for r in manifest}, 'Incomplete OOF cohort coverage')
    all_test_ids = []
    folds = []
    for k in range(5):
        folder = evidence / f'cv5/fold_{k}'
        rotated = read_csv(folder / 'manifest.csv')
        require({r['patient_id'] for r in rotated} == set(oof_map), 'Fold cohort mismatch')
        for r in rotated:
            fold = int(r['fold'])
            expected = 'test' if fold == k else 'val' if fold == (k + 1) % 5 else 'train'
            require(r['split'] == expected, f'Incorrect rotation in fold {k}')
        saved, predictions = check_run(folder, rotated)
        check_metrics(predictions, cv['fold_metrics'][k], f'aggregate fold {k}')
        for r in predictions:
            pooled = oof_map[r['patient_id']]
            require(int(r['label']) == int(pooled['label']), 'OOF label mismatch')
            same(r['probability_hgg'], pooled['probability_hgg'], 'OOF score correspondence')
        all_test_ids.extend(r['patient_id'] for r in predictions)
        folds.append(saved)
    require(len(all_test_ids) == len(set(all_test_ids)) == 369, 'Test folds do not cover each patient exactly once')
    for key, saved in cv['fold_summary'].items():
        scores = [float(f[key]) for f in folds]
        same(statistics.mean(scores), saved['mean'], f'mean fold {key}')
        same(statistics.stdev(scores), saved['std'], f'sample fold SD {key}')
    debug = read_json(evidence / 'overfit_check/overfit_check.json')
    require(debug['passed'] and debug['checkpoint_reload_predictions_match'], 'Debug check failed')
    require(len(debug['patient_ids']) == debug['n_patients'] == 8 and set(debug['patient_ids']) <= parts['train'],
            'Debug patients are not a training-only subset')
    record = read_json(root / 'results/provenance/source_record.json')
    hashes = record.get('module_sha256_normalized_lf', record['module_sha256'])
    for rel, expected in hashes.items():
        content = (root / rel).read_bytes()
        if 'module_sha256_normalized_lf' in record:
            content = content.replace(b'\r\n', b'\n')
        require(hashlib.sha256(content).hexdigest() == expected, f'Executed source changed: {rel}')
    for rel, expected in record['documents'].items():
        require(hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected, f'Document changed: {rel}')
    return {'passed': True, 'patients': 369, 'main_runs_checked': 7, 'oof_predictions': len(oof),
            'metrics_recomputed': True, 'partitions_disjoint': True, 'training_source_hashes_match': True,
            'pooled_balanced_accuracy': cv['balanced_accuracy'], 'fixed_balanced_accuracy': fixed['balanced_accuracy'],
            'whole_brain_balanced_accuracy': brain['balanced_accuracy'],
            'claim_boundary': 'Saved-artifact arithmetic and consistency checks; no GPU retraining or checkpoint loading.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
