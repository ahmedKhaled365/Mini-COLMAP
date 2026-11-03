"""Print a comparison table of Mini-COLMAP run folders.

Usage (from the results folder):
    python compare_runs.py RUN_DIR [RUN_DIR ...] [--markdown]

Reads report.txt, registration.csv and (if present) ba_log.csv of each run.
"""
import csv
import os
import re
import statistics
import sys


def field(text, pattern, default='-'):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def summarize(run):
    text = open(os.path.join(run, 'report.txt')).read()
    reg = list(csv.DictReader(open(os.path.join(run, 'registration.csv'))))
    ok = [r for r in reg if r['registered'] == '1']
    landmarks = int(field(text, r'Landmarks: (\d+)', '0'))
    two = int(field(text, r'2-view (\d+)', '0'))

    row = {
        'run': os.path.basename(os.path.normpath(run)),
        'reg': field(text, r'Registered: (\d+)'),
        'landmarks': landmarks,
        'reliable': field(text, r'Reliable landmarks: (\d+)'),
        'object': field(text, r'Object region: (\d+)'),
        'object_rel': field(text, r'\((\d+) reliable'),
        'track': field(text, r'Track length: mean ([\d.]+)'),
        'two_view_pct': f'{100.0 * two / max(1, landmarks):.1f}',
        'angle_med': field(text, r'Tri\. angle \[deg\]: mean [\d.]+ \| median ([\d.]+)'),
        'rp_med': field(text, r'Reprojection \[px\]: mean [\d.]+ \| median ([\d.]+)'),
        'rp_p95': field(text, r'Reprojection \[px\]: .*?P95 ([\d.]+)'),
        'pnp_mean': f"{statistics.mean(float(r['inlier_ratio']) for r in ok):.3f}" if ok else '-',
        'pnp_min': f"{min(float(r['inlier_ratio']) for r in ok):.3f}" if ok else '-',
        'step_cv': field(text, r'Camera step CV: ([\d.]+)'),
        'focal': field(text, r'final f ([\d.]+)', field(text, r'focal: ([\d.]+)')),
        'k1': field(text, r'final k1 (-?[\d.e-]+)', '0'),
    }
    return row


COLUMNS = ['run', 'reg', 'landmarks', 'reliable', 'object', 'object_rel', 'track',
           'two_view_pct', 'angle_med', 'rp_med', 'rp_p95', 'pnp_mean', 'pnp_min',
           'step_cv', 'focal', 'k1']


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    markdown = '--markdown' in sys.argv
    rows = [summarize(r) for r in args]

    if markdown:
        print('| ' + ' | '.join(COLUMNS) + ' |')
        print('|' + '---|' * len(COLUMNS))
        for r in rows:
            print('| ' + ' | '.join(str(r[c]) for c in COLUMNS) + ' |')
        return

    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLUMNS}
    print('  '.join(c.rjust(widths[c]) for c in COLUMNS))
    for r in rows:
        print('  '.join(str(r[c]).rjust(widths[c]) for c in COLUMNS))


if __name__ == '__main__':
    main()
