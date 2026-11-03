import csv
import sys
from pathlib import Path


def main(path: str) -> None:
    rows = list(csv.DictReader(Path(path).open(newline='', encoding='utf-8-sig')))
    if not rows:
        raise SystemExit('CSV has no data rows')
    required = {'actual_g', 'predicted_g'}
    if not required.issubset(rows[0]):
        raise SystemExit('CSV needs columns: actual_g,predicted_g (ingredient_name is optional)')
    errors = []
    pct_errors = []
    for row in rows:
        actual = float(row['actual_g'])
        predicted = float(row['predicted_g'])
        errors.append(abs(predicted - actual))
        if actual != 0:
            pct_errors.append(abs(predicted - actual) / abs(actual) * 100)
    print(f'Count: {len(errors)}')
    print(f'MAE: {sum(errors) / len(errors):.2f} g')
    if pct_errors:
        print(f'MAPE: {sum(pct_errors) / len(pct_errors):.2f}% (actual != 0 only)')
    else:
        print('MAPE: undefined (all actual values are zero)')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python evaluate_weights.py weighed_results.csv')
    main(sys.argv[1])
