import os
import pandas as pd
from datetime import time

PERM_FILE = 'data/permanent_6yr.csv'
MID_FILE = 'data/midblock_6yr.csv'
OUT_DIR = 'data'
WINDOW_START = time(8, 0, 0)
WINDOW_END = time(8, 30, 0)
FREQ_MINUTES = 15
SLOTS_PER_WINDOW = 3


def load():
    perm = pd.read_csv(PERM_FILE, parse_dates=['DateTime'])
    mid = pd.read_csv(MID_FILE, parse_dates=['DateTime'])
    return perm, mid


def report_span(perm, mid):
    start = min(perm['DateTime'].min(), mid['DateTime'].min())
    end = max(perm['DateTime'].max(), mid['DateTime'].max())
    years = (end - start).days / 365.25
    print('Data span')
    print(f"  {start.date()} -> {end.date()}  ({years:.2f} years)")
    print(f"  permanent stations : {perm['Site'].nunique()}")
    print(f"  midblock locations : {mid['Location Description'].nunique()}")
    return start, end


def report_nulls(name, df):
    nulls = df.isna().sum()
    total = int(nulls.sum())
    if total == 0:
        print(f"  {name:10s} no nulls in {len(df):,} rows "
              f"(prepare_data.py drops incomplete rows upstream)")
    else:
        print(f"  {name:10s} {total:,} nulls")
        for col, n in nulls[nulls > 0].items():
            print(f"      {col}: {n:,} ({100 * n / len(df):.2f}%)")


def expected_slots(start, end, minutes):
    if end <= start:
        return 0
    return int((end - start).total_seconds() // (minutes * 60)) + 1


def expected_window_slots(start, end):
    if end <= start:
        return 0
    days = pd.date_range(start.normalize(), end.normalize(), freq='D')
    weekdays = days[days.weekday < 5]
    return len(weekdays) * SLOTS_PER_WINDOW


def permanent_by_year(perm, span_start, span_end):
    rows = []
    # filter to the morning window on weekdays
    window = perm[
        (perm['DateTime'].dt.time >= WINDOW_START)
        & (perm['DateTime'].dt.time <= WINDOW_END)
        & (perm['DateTime'].dt.weekday < 5)
    ]
    window_counts = window.groupby(['Site', window['DateTime'].dt.year]).size()

    for site, g in perm.groupby('Site'):
        site_start = max(g['DateTime'].min(), span_start)
        site_end = min(g['DateTime'].max(), span_end)
        for year in range(span_start.year, span_end.year + 1):
            y0 = max(pd.Timestamp(year=year, month=1, day=1), site_start)
            y1 = min(pd.Timestamp(year=year, month=12, day=31, hour=23, minute=45), site_end)
            expected = expected_slots(y0, y1, FREQ_MINUTES)
            if expected == 0:
                continue
            observed = int((g['DateTime'].dt.year == year).sum())
            exp_win = expected_window_slots(y0, y1)
            obs_win = int(window_counts.get((site, year), 0))
            rows.append({
                'site': site,
                'year': year,
                'expected': expected,
                'observed': observed,
                'missing': expected - observed,
                'missing_pct': round(100 * (expected - observed) / expected, 2),
                'window_expected': exp_win,
                'window_observed': obs_win,
                'window_missing_pct': round(100 * (exp_win - obs_win) / exp_win, 2) if exp_win else 0.0,
            })
    return pd.DataFrame(rows)


def permanent_gaps(perm, span_end):
    step = pd.Timedelta(minutes=FREQ_MINUTES)
    rows = []
    for site, g in perm.groupby('Site'):
        stamps = g['DateTime'].sort_values()
        diffs = stamps.diff()
        gaps = diffs[diffs > step]
        longest = gaps.max()
        rows.append({
            'site': site,
            'first_reading': stamps.min(),
            'last_reading': stamps.max(),
            'readings': len(stamps),
            'gap_count': int((gaps > step).sum()),
            'gaps_over_1day': int((gaps > pd.Timedelta(days=1)).sum()),
            'longest_gap_hours': round(longest.total_seconds() / 3600, 1) if pd.notna(longest) else 0.0,
            'missing_slots': int(sum((gaps / step).round()) - len(gaps)) if len(gaps) else 0,
            'days_silent_at_end': (span_end - stamps.max()).days,
        })
    return pd.DataFrame(rows).sort_values('missing_slots', ascending=False)


def midblock_coverage(mid, span_end):
    rows = []
    for loc, g in mid.groupby('Location Description'):
        stamps = g['DateTime']
        rows.append({
            'location': loc,
            'readings': len(g),
            'survey_days': stamps.dt.date.nunique(),
            'first_reading': stamps.min(),
            'last_reading': stamps.max(),
            'years_since_last': round((span_end - stamps.max()).days / 365.25, 2),
        })
    return pd.DataFrame(rows).sort_values('survey_days')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    perm, mid = load()

    span_start, span_end = report_span(perm, mid)

    print('\nNull values')
    report_nulls('permanent', perm)
    report_nulls('midblock', mid)

    by_year = permanent_by_year(perm, span_start, span_end)
    gaps = permanent_gaps(perm, span_end)
    coverage = midblock_coverage(mid, span_end)

    print('\nPermanent stations, missing 15-minute readings by year')
    yearly = by_year.groupby('year').agg(
        expected=('expected', 'sum'),
        observed=('observed', 'sum'),
        missing=('missing', 'sum'),
    )
    yearly['missing_pct'] = (100 * yearly['missing'] / yearly['expected']).round(2)
    print(yearly.to_string())

    print('\nPermanent stations, missing readings in the 08:00-08:30 weekday window')
    win = by_year.groupby('year').agg(
        expected=('window_expected', 'sum'),
        observed=('window_observed', 'sum'),
    )
    win['missing'] = win['expected'] - win['observed']
    win['missing_pct'] = (100 * win['missing'] / win['expected']).round(2)
    print(win.to_string())

    print('\nWorst stations by missing slots')
    print(gaps[['site', 'readings', 'gap_count', 'gaps_over_1day',
                'longest_gap_hours', 'missing_slots']].head(10).to_string(index=False))

    print('\nMidblock survey coverage')
    print(f"  locations              : {len(coverage):,}")
    print(f"  median survey days     : {coverage['survey_days'].median():.0f}")
    print(f"  surveyed on one day    : {(coverage['survey_days'] == 1).sum():,}")
    print(f"  not surveyed in 3+ yrs : {(coverage['years_since_last'] >= 3).sum():,}")

    by_year.to_csv(f'{OUT_DIR}/missing_permanent_by_year.csv', index=False)
    gaps.to_csv(f'{OUT_DIR}/missing_permanent_sites.csv', index=False)
    coverage.to_csv(f'{OUT_DIR}/midblock_coverage.csv', index=False)
    print(f"\nSaved -> {OUT_DIR}/missing_permanent_by_year.csv")
    print(f"Saved -> {OUT_DIR}/missing_permanent_sites.csv")
    print(f"Saved -> {OUT_DIR}/midblock_coverage.csv")


if __name__ == '__main__':
    main()
