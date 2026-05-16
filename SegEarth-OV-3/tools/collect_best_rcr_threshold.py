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
    parser.add_argument(
        '--work-dir-root',
        default='work_dirs',
        help='Fallback root containing cfg_*_rcr/results.txt files.')
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


def parse_class_table(text):
    rows = []
    in_table = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if 'per class results:' in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith('+'):
            continue
        if not line.startswith('|'):
            if rows:
                break
            continue
        columns = [item.strip() for item in line.strip('|').split('|')]
        if len(columns) != 3 or columns[0] == 'Class':
            continue
        try:
            rows.append({
                'Class': columns[0],
                'IoU': float(columns[1]),
                'Acc': float(columns[2]),
            })
        except ValueError:
            continue
    return rows


def write_classwise_files(run_dir, summary, class_rows):
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    csv_path = run_path / 'class_iou_results.csv'
    json_path = run_path / 'class_iou_results.json'
    txt_path = run_path / 'class_iou_results.txt'

    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Class', 'IoU', 'Acc'])
        writer.writeheader()
        writer.writerows(class_rows)

    with json_path.open('w') as handle:
        json.dump({
            'aAcc': summary.get('aAcc'),
            'mIoU': summary.get('mIoU'),
            'mAcc': summary.get('mAcc'),
            'classes': class_rows,
        }, handle, ensure_ascii=False, indent=2)

    with txt_path.open('w') as handle:
        handle.write('Class\tIoU\tAcc\n')
        for row in class_rows:
            handle.write(f"{row['Class']}\t{row['IoU']}\t{row['Acc']}\n")
        handle.write(f"\naAcc\t{summary.get('aAcc')}\n")
        handle.write(f"mIoU\t{summary.get('mIoU')}\n")
        handle.write(f"mAcc\t{summary.get('mAcc')}\n")


def parse_log_result(log_path):
    text = log_path.read_text(errors='ignore')
    output_matches = re.findall(r"output_dir='([^']+)'", text)
    metric_matches = re.findall(
        r"Iter\(test\).*?aAcc:\s*([0-9.]+)\s+"
        r"mIoU:\s*([0-9.]+)\s+mAcc:\s*([0-9.]+)",
        text,
    )
    if not output_matches or not metric_matches:
        return None

    output_dir = output_matches[-1]
    run_info = parse_run_name(Path(output_dir).name)
    if run_info is None:
        return None

    a_acc, miou, m_acc = metric_matches[-1]
    class_rows = parse_class_table(text)
    summary = {
        'aAcc': float(a_acc),
        'mIoU': float(miou),
        'mAcc': float(m_acc),
    }
    if class_rows:
        write_classwise_files(output_dir, summary, class_rows)

    return {
        **run_info,
        'run_dir': output_dir,
        **summary,
        'classes': class_rows,
        'source': str(log_path),
        'finished_at': log_path.stat().st_mtime,
    }


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    if root.exists():
        root_candidates = sorted(path for path in root.iterdir() if path.is_dir())
    else:
        root_candidates = []
    for run_dir in root_candidates:
        run_info = parse_run_name(run_dir.name)
        if run_info is not None:
            run_dirs.append((run_dir, run_info))

    rows_by_run_dir = {}
    for result_path in sorted(root.glob('*/class_iou_results.json')):
        run_info = parse_run_name(result_path.parent.name)
        if run_info is None:
            continue

        with result_path.open() as handle:
            result = json.load(handle)

        rows_by_run_dir[str(result_path.parent)] = {
            **run_info,
            'run_dir': str(result_path.parent),
            'aAcc': result.get('aAcc'),
            'mIoU': result.get('mIoU'),
            'mAcc': result.get('mAcc'),
            'classes': result.get('classes', []),
            'source': 'class_iou_results.json',
            'finished_at': result_path.stat().st_mtime,
        }

    work_dir_root = Path(args.work_dir_root)
    for log_path in sorted(work_dir_root.glob('cfg_*_rcr/20*/*.log')):
        row = parse_log_result(log_path)
        if row is None:
            continue
        run_dir = row['run_dir']
        if not str(run_dir).startswith(str(root)):
            continue
        current = rows_by_run_dir.get(run_dir)
        if current is None or row['finished_at'] > current.get('finished_at', 0):
            rows_by_run_dir[run_dir] = row

    rows = sorted(rows_by_run_dir.values(), key=lambda item: (item['dataset'], item['confidence_threshold'], item['prob_thd']))

    if not rows:
        raise SystemExit(
            f'No completed RCR threshold results found under {root}. '
            'Run at least one threshold setting first.')

    best_by_dataset = {}
    for row in rows:
        dataset = row['dataset']
        current = best_by_dataset.get(dataset)
        if current is None or float(row['mIoU']) > float(current['mIoU']):
            best_by_dataset[dataset] = row

    all_csv = out_dir / 'all_rcr_threshold_results.csv'
    best_csv = out_dir / 'best_rcr_threshold_results.csv'
    best_json = out_dir / 'best_rcr_threshold_results.json'
    all_class_csv = out_dir / 'all_rcr_class_iou_results.csv'
    best_class_csv = out_dir / 'best_rcr_class_iou_results.csv'

    fieldnames = [
        'dataset',
        'confidence_threshold',
        'prob_thd',
        'aAcc',
        'mIoU',
        'mAcc',
        'source',
        'finished_at',
        'run_dir',
    ]
    with all_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    best_rows = [best_by_dataset[key] for key in sorted(best_by_dataset)]
    with best_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(best_rows)

    with best_json.open('w') as handle:
        json.dump(best_by_dataset, handle, indent=2)

    class_fieldnames = [
        'dataset',
        'confidence_threshold',
        'prob_thd',
        'Class',
        'IoU',
        'Acc',
        'run_dir',
        'source',
    ]
    all_class_rows = []
    for row in rows:
        for class_row in row.get('classes', []):
            all_class_rows.append({
                'dataset': row['dataset'],
                'confidence_threshold': row['confidence_threshold'],
                'prob_thd': row['prob_thd'],
                'Class': class_row.get('Class'),
                'IoU': class_row.get('IoU'),
                'Acc': class_row.get('Acc'),
                'run_dir': row['run_dir'],
                'source': row['source'],
            })
    best_class_rows = []
    for row in best_rows:
        for class_row in row.get('classes', []):
            best_class_rows.append({
                'dataset': row['dataset'],
                'confidence_threshold': row['confidence_threshold'],
                'prob_thd': row['prob_thd'],
                'Class': class_row.get('Class'),
                'IoU': class_row.get('IoU'),
                'Acc': class_row.get('Acc'),
                'run_dir': row['run_dir'],
                'source': row['source'],
            })

    with all_class_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fieldnames)
        writer.writeheader()
        writer.writerows(all_class_rows)

    with best_class_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fieldnames)
        writer.writeheader()
        writer.writerows(best_class_rows)

    print(f'Saved all results: {all_csv}')
    print(f'Saved best results: {best_csv}')
    print(f'Saved best results: {best_json}')
    print(f'Saved all class IoU results: {all_class_csv}')
    print(f'Saved best class IoU results: {best_class_csv}')
    print(f'Collected {len(rows)} finished runs from {len(best_rows)} dataset(s).')
    for row in best_rows:
        print(
            f"{row['dataset']}: mIoU={row['mIoU']} "
            f"ct={row['confidence_threshold']} pt={row['prob_thd']} "
            f"source={row['source']} dir={row['run_dir']}")


if __name__ == '__main__':
    main()
