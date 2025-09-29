#!/usr/bin/env python3
# Huber robust regression using SciPy's least_squares
# Students: complete the TODO sections

import csv
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import re

# --- Read CSV ---
def read_csv_data(file_path):
    """
    TODO:
    - Open the CSV file with `with open(file_path, mode='r', newline='')`
    - Create a csv.reader object
    - Skip the header row
    - For each row:
        * Extract Year (int), Team (str), League (str), Player (str), Salary (float)
        * Append as a dictionary to a list
    - Return the list of dictionaries
    """

    #Rows is being created and initialized to an empty list. 
    rows: List[Dict[str, object]]= []
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        header = next(reader, None)
        for r in reader:
            year = int(r[0])
            team = r[1].strip()
            league = r[2].strip()
            player = r[3].strip()
            salary = float(r[4])
            rows.append({'Year': year, 'Team': team, 'League': league, 'Player': player, 'Salary': salary})
    return rows



def scan_csv_issues(file_path, expected_cols=None, require_space_after_comma=False):
    """Scan a CSV file for simple issues.

    Checks performed:
      - column count mismatch (compared to header or expected_cols)
      - missing data (empty field after stripping)
      - missing space after comma (flagged as 'suspicious' unless require_space_after_comma=True)

    Returns a list of issue dicts:
      { 'line': int, 'type': str, 'detail': str, ... }
    """
    issues = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return issues

    # Determine expected column count from header if not provided
    try:
        header_fields = next(csv.reader([lines[0]]))
        header_cols = len(header_fields)
    except Exception:
        header_cols = expected_cols or 5

    exp_cols = expected_cols if expected_cols is not None else header_cols

    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip('\n')
        if line.strip() == '':
            issues.append({'line': lineno, 'type': 'empty_line', 'detail': 'blank or empty line'})
            continue

        # Try parsing using csv.reader so quoted fields are handled
        try:
            fields = next(csv.reader([line]))
        except Exception as e:
            issues.append({'line': lineno, 'type': 'parse_error', 'detail': str(e), 'raw': line})
            continue

        # Column count check
        if len(fields) != exp_cols:
            issues.append({
                'line': lineno,
                'type': 'cols_mismatch',
                'expected': exp_cols,
                'found': len(fields),
                'raw': line
            })

        # Missing data check (empty fields)
        for col_idx, val in enumerate(fields):
            if val.strip() == '':
                issues.append({
                    'line': lineno,
                    'type': 'missing_data',
                    'col': col_idx,
                    'raw': line
                })

        # Missing-space-after-comma detection: many CSVs do not require a space after comma,
        # but this helps spot likely manual formatting errors such as 'Doe,John' where a space is expected.
        # The regex finds a comma followed immediately by a non-space, non-quote character.
        m = re.search(r',(?=[^"\s])', line)
        if m:
            kind = 'missing_space_after_comma' if require_space_after_comma else 'suspicious_missing_space'
            issues.append({'line': lineno, 'type': kind, 'raw': line})

    return issues

data = read_csv_data('Salaries.csv')
if not data:
    raise SystemExit("No data loaded from Salaries.csv")

# --- Load arrays from rows ---
# TODO: Read values into arrays using np.array
x = np.array([row['Year'] for row in data], dtype=np.float64)
y = np.array([row['Salary'] for row in data], dtype=np.float64)

# --- Center x to improve conditioning ---
# TODO: compute xm = mean of x; xc = x - xm
xm = np.mean(x)
xc = x - xm

# --- OLS initialization on centered x ---
# TODO: build Xc = [1, xc] and solve least squares for ac0 (centered intercept) and b0 (slope)
Xc = np.column_stack([np.ones_like(xc), xc])
beta_ols = np.linalg.lstsq(Xc, y, rcond=None)[0]
ac0, b0 = beta_ols

# --- Residual function for least_squares (given) ---
def resid_centered(p):
    """
    Residuals for robust regression:
      r = y - (a_c + b * x_c)
    where a_c is the intercept in centered coordinates.
    """
    ac, b = p
    return y - (ac + b * xc)

# --- Robust scale estimate from OLS residuals (given) ---
r0 = resid_centered((ac0, b0))
sigma_mad = stats.median_abs_deviation(r0, scale='normal')
if sigma_mad == 0:
    sigma_mad = np.std(r0) or 1.0

# --- Huber threshold (delta = c * sigma_mad) ---
# TODO: choose c (≈1.345 is standard) and compute delta
c = 1.345       # TODO
delta = c * sigma_mad

# --- Run robust regression with SciPy ---
# TODO: call scipy.optimize.least_squares with:
#   - resid_centered as residual function
#   - x0 = np.array([ac0, b0], dtype=np.float64)
#   - loss='huber', f_scale=delta
#   - method='trf', x_scale='jac'
#   - and reasonable tolerances (ftol, xtol, gtol)
res = least_squares(
    resid_centered,
    x0=np.array([ac0, b0], dtype=np.float64),
    loss='huber', f_scale=delta,
    method='trf', x_scale='jac',
    ftol=1e-10, xtol=1e-10, gtol=1e-10
)

# --- Extract solution (centered), then un-center intercept ---
ac_h, b_h = res.x     # TODO: res.x
a_h = ac_h - b_h * xm  # TODO: ac_h - b_h * xm

# --- Final reporting ---
r_final = y - (ac_h + b_h * xc)      # TODO: y - (ac_h + b_h * xc)
sigma_final = stats.median_abs_deviation(r_final, scale='normal')  # TODO: stats.median_abs_deviation(r_final, scale='normal')

print("=== SciPy Huber Regression (least_squares, centered x) ===")
print(f"Slope (robust):     {b_h}")
print(f"Intercept (robust): {a_h}")
print(f"Scale (final MAD):  {sigma_final}")
print(f"Function evals:     {res.nfev if res is not None else 'TODO'}")

# --- OLS for reference (optional, in original coordinates) ---
a_ols = (ac0 - b0 * xm)   # TODO: (ac0 - b0 * xm)
b_ols = b0   # TODO: b0

# --- Plot results ---
# (These will work after the TODOs above are filled.)
x_line = np.linspace(np.min(x) if x is not None else 0,
                     np.max(x) if x is not None else 1, 200)
y_line_huber = a_h + b_h * (x_line - xm)  # TODO:
y_line_ols = a_ols + b_ols * x_line  # TODO:

plt.figure(figsize=(8,5))
plt.scatter(x, y, s=16, alpha=0.6, label='Data')
#plt.plot(x_line, y_line_huber, lw=2, label='Huber (SciPy, centered x)')
plt.plot(x_line, y_line_ols, lw=2, color='red', ls='--', label='OLS (reference)')
plt.xlabel('Year')
plt.ylabel('Salary [$]')
plt.title('Salary vs Year — Huber Robust Regression (SciPy least_squares)')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('salaries_huber_fit_scipy_centered.png', dpi=150)
plt.show()
plt.legend(title="Regression")