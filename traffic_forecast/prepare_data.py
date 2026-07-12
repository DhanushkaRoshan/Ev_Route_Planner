import os
import pandas as pd
from datetime import datetime

# filter out anything before this date
CUTOFF = datetime(2020, 5, 14)
os.makedirs('data', exist_ok=True)

# handle permanent count stations first
print("Processing permanent count stations...")
perm = pd.read_csv('data/Permanent_Count_Station_Traffic_Counts.csv')
perm['DateTime'] = pd.to_datetime(perm['Timestamp'], errors='coerce')
perm = perm[perm['DateTime'] >= CUTOFF].copy()
perm['Count'] = pd.to_numeric(perm['Total'], errors='coerce')

cols_to_keep = ['DateTime', 'Count', 'Latitude', 'Longitude', 'Site']
perm = perm[cols_to_keep].dropna(
    subset=['Latitude', 'Longitude', 'Count']
).reset_index(drop=True)
perm.to_csv('data/permanent_6yr.csv', index=False)
print(f"  saved {len(perm):,} rows -> data/permanent_6yr.csv")

# load coordinate lookup for midblock stations
print("Loading coordinate lookup...")
coords = pd.read_csv('data/locations_with_coordinates.csv')
coords = coords[coords['Coordinates_1'].notna()].copy()
coords[['Latitude', 'Longitude']] = coords['Coordinates_1'].str.split(',', expand=True).astype(float)
coords = coords[['Locations', 'Latitude', 'Longitude']].rename(
    columns={'Locations': 'Location Description'}
)
print(f"  {len(coords):,} locations with coordinates")

# midblock traffic counts - big file so read in chunks
print("Processing midblock traffic counts (chunked read)...")
DATE_FMT = '%Y %b %d %I:%M:%S %p'
chunks = []
for chunk in pd.read_csv('data/Midblock_Traffic_Counts.csv', chunksize=500_000,
                          dtype={'Count 15 Minutes': str}):
    chunk['DateTime'] = pd.to_datetime(chunk['Count Date'], format=DATE_FMT, errors='coerce')
    chunk = chunk[chunk['DateTime'] >= CUTOFF]
    if len(chunk):
        chunks.append(chunk)

if chunks:
    mid = pd.concat(chunks, ignore_index=True)
else:
    mid = pd.DataFrame(columns=['DateTime', 'Count 15 Minutes', 'Location Description'])

print(f"  {len(mid):,} rows after date filter")

# merge with coords and clean up
mid = mid.merge(coords, on='Location Description', how='left')
mid = mid.dropna(subset=['Latitude', 'Longitude'])
mid['Count'] = pd.to_numeric(mid['Count 15 Minutes'], errors='coerce')
mid = mid[['DateTime', 'Count', 'Latitude', 'Longitude', 'Location Description']].dropna().reset_index(drop=True)
mid.to_csv('data/midblock_6yr.csv', index=False)
print(f"  saved {len(mid):,} rows -> data/midblock_6yr.csv")

print("\nDone.")
print(f"  Permanent: {perm['DateTime'].min()} -> {perm['DateTime'].max()}")
if len(mid):
    print(f"  Midblock:  {mid['DateTime'].min()} -> {mid['DateTime'].max()}")
