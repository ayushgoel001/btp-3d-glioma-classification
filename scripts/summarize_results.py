"""Generate the human-readable experiment record from frozen metric JSON files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def generate():
    gpu = ROOT / 'results/gpu'
    roi = load(gpu / 'fixed_split/test_metrics.json')
    brain = load(gpu / 'whole_brain_fixed/test_metrics.json')
    cv = load(gpu / 'cv5/cross_validation_metrics.json')
    debug = load(gpu / 'overfit_check/overfit_check.json')
    keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'sensitivity_hgg', 'specificity_lgg']
    lines = ['# Executed GPU results', '',
             ('These values come from the saved real-data Colab experiment, using all 369 labelled BraTS 2020 patients. '
              'No new GPU training was performed while preparing this repository.'), '',
             ('Main models use four MRI channels, 96-cubed inputs, base width 16 and seed 42. '
              'The effective objective is unweighted cross-entropy because configured weights cancel at physical batch one.'), '',
             '## Aggregate results', '',
             '| Experiment | n | Accuracy | Balanced accuracy | Macro-F1 | HGG recall | LGG recall | AUC |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, m in [('ROI fixed split', roi), ('ROI pooled OOF', cv), ('Whole-brain fixed split', brain)]:
        cells = [name, str(m['n_patients'])] + [f'{100*m[k]:.2f}%' for k in keys] + [f'{m["roc_auc"]:.4f}']
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', ('Fixed ROI and whole-brain results refer to the same 56 patients. '
                   'The pooled OOF result combines one held-out prediction per cohort patient from five fitted models.'), '',
              '## Fold results', '',
              '| Fold | n | Accuracy | Balanced accuracy | Macro-F1 | HGG recall | LGG recall | AUC |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in cv['fold_metrics']:
        cells = [str(m['fold']), str(m['n_patients'])] + [f'{100*m[k]:.2f}%' for k in keys] + [f'{m["roc_auc"]:.4f}']
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Mean fold scores', '', '| Metric | Mean | Sample SD |', '| --- | ---: | ---: |']
    for key, label in [('balanced_accuracy', 'Balanced accuracy'), ('macro_f1', 'Macro-F1'),
                       ('sensitivity_hgg', 'HGG recall'), ('specificity_lgg', 'LGG recall'), ('roc_auc', 'ROC-AUC')]:
        m = cv['fold_summary'][key]
        values = [f'{m["mean"]:.4f}', f'{m["std"]:.4f}'] if key == 'roc_auc' else [f'{100*m["mean"]:.2f}%', f'{100*m["std"]:.2f} pp']
        lines.append('| ' + ' | '.join([label] + values) + ' |')
    lines += ['', ('Sample SD describes the five partitions and uses denominator four. '
              'It is not a confidence interval. Pooled AUC 0.8815 differs from mean fold AUC 0.9008 '
              'because pooled ranking compares scores across different fitted models.'), '',
              '## Pooled confusion counts', '',
              '| True class | Predicted LGG | Predicted HGG | Total |', '| --- | ---: | ---: | ---: |']
    for name, row in zip(['LGG', 'HGG'], cv['confusion_matrix_lgg_hgg']):
        lines.append(f'| {name} | {row[0]} | {row[1]} | {sum(row)} |')
    lines += ['', ('The pooled predictions contain 15 LGG-to-HGG errors and 32 HGG-to-LGG errors. '
              'Correct predictions total 322 of 369.'), '', '## Training durations and selection', '',
              '| Model | Selected epoch | Stopped epoch | Training minutes |', '| --- | ---: | ---: | ---: |']
    for name, m, folder in [('ROI fixed', roi, gpu/'fixed_split')] + [
        (f'CV fold {f["fold"]}', f, gpu/f'cv5/fold_{f["fold"]}') for f in cv['fold_metrics']
    ] + [('Whole-brain fixed', brain, gpu/'whole_brain_fixed')]:
        stopped = len((folder/'history.csv').read_text().strip().splitlines()) - 1
        lines.append(f'| {name} | {m["best_epoch"]} | {stopped} | {m["elapsed_seconds"]/60:.2f} |')
    total = sum(f['elapsed_seconds'] for f in cv['fold_metrics'])
    lines += ['', (f'The five ROI fits total {total:.2f} seconds ({total/60:.2f} minutes). '
              'The saved Tesla T4 timers exclude download, preprocessing and document preparation.'), '',
              '## Debugging check', '',
              (f'The training-only eight-patient memorization check passed after {debug["epochs"]} epochs, '
              f'with training accuracy {debug["training_accuracy"]:.1%}, loss {debug["final_loss"]:.5f} '
              'and matching predictions after checkpoint reload. It uses width four and dropout zero. '
              'It is not held-out evaluation.'), '', '## Evidence', '',
              '- [Compact GPU export](gpu/)',
              '- [Pooled patient predictions](gpu/cv5/out_of_fold_predictions.csv)',
              '- [Source and document hashes](provenance/source_record.json)',
              '- [Original executed notebook](../notebooks/BTP1_GPU_Executed.ipynb)',
              '- [Implementation limits](../docs/KNOWN_LIMITATIONS.md)', '',
              ('Raw MRI, cache tensors, checkpoints and the full results ZIP are excluded from Git. '
              'The original local export retains the checkpoints. The repository evidence supports arithmetic '
              'and partition checks without independently repeating GPU training.'), '']
    target = ROOT / 'results/EXECUTED_RESULTS.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    return target


if __name__ == '__main__':
    print(generate())
