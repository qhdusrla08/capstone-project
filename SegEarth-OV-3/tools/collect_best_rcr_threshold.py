import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect best RCR threshold sweep results.')
    parser.add_argument(
        '--root',
        default='outputs/rcr_threshold_sweep',
        help='Root directory that contains per-threshold evaluation outputs.')
    parser.add_argument(
        '--out',
        default='outputs/rcr_threshold_sweep/best_rcr_thresholds',
        help='Directory to save best-threshold summaries.')
    return parser.parse_args()


def parse_run_name(name):
    pattern = re.compile(
        r'^(?P<dataset>.+)_ct(?P<ct>[0-9.]+)_pt(?P<pt>[0-9.]+)$')
    match = pattern.match(name)
    if match is None:
        return None
    return {
        'dataset': match.group('dataset'),
        'confidence_threshold': float(match.group('ct')),
        'prob_thd': float(match.group('pt')),
    }


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for result_path in sorted(root.glob('*/class_iou_results.json')):
        run_info = parse_run_name(result_path.parent.name)
        if run_info is None:
            continue

        with result_path.open() as handle:
            result = json.load(handle)

        rows.append({
            **run_info,
            'run_dir': str(result_path.parent),
            'aAcc': result.get('aAcc'),
            'mIoU': result.get('mIoU'),
            'mAcc': result.get('mAcc'),
        })

    if not rows:
        raise SystemExit(
            f'No class_iou_results.json files found under {root}. '
            'Run the threshold sweep first.')

    best_by_dataset = {}
    for row in rows:
        dataset = row['dataset']
        current = best_by_dataset.get(dataset)
        if current is None or float(row['mIoU']) > float(current['mIoU']):
            best_by_dataset[dataset] = row

    all_csv = out_dir / 'all_rcr_threshold_results.csv'
    best_csv = out_dir / 'best_rcr_threshold_results.csv'
    best_json = out_dir / 'best_rcr_threshold_results.json'

    fieldnames = [
        'dataset',
        'confidence_threshold',
        'prob_thd',
        'aAcc',
        'mIoU',
        'mAcc',
        'run_dir',
    ]
    with all_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_rows = [best_by_dataset[key] for key in sorted(best_by_dataset)]
    with best_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    with best_json.open('w') as handle:
        json.dump(best_by_dataset, handle, indent=2)

    print(f'Saved all results: {all_csv}')
    print(f'Saved best results: {best_csv}')
    print(f'Saved best results: {best_json}')
    for row in best_rows:
        print(
            f"{row['dataset']}: mIoU={row['mIoU']} "
            f"ct={row['confidence_threshold']} pt={row['prob_thd']} "
            f"dir={row['run_dir']}")


if __name__ == '__main__':
    main()
