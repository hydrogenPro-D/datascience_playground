import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
metadata_df = pd.read_csv("data/metadata.csv")
excel_file_path = r"QC progress overview - Line 1 COPY2.xlsm"

# Helper function to extract number from items like "B123"
def extract_number(item):
    try:
        return int(item[1:])  # Extract everything after the first character
    except (ValueError, IndexError):
        return float('inf')  # Put non-numeric items at the end

# ============================================================================
# RECREATE electrode_analysis_df (from the notebook)
# ============================================================================

# Load released_qc
excel_file = pd.ExcelFile(excel_file_path)
sheet_names = excel_file.sheet_names
overview_sheets = [name for name in sheet_names if name.startswith("Overview")][:-1]

released_qc = []
for sheet in overview_sheets:
    df = pd.read_excel(excel_file_path, sheet_name=sheet, header=None)
    if len(df.columns) > 11:
        col_k = df.iloc[:, 10]
        col_l = df.iloc[:, 11]
        release_header = col_k.iloc[0]
        if release_header == "Release":
            col_k_values = col_k.iloc[3:].dropna().tolist()
            col_l_values = col_l.iloc[3:].dropna().tolist()
            combined_list = col_k_values + col_l_values
            released_qc.extend(combined_list)

released_qc_cleaned = sorted(list(set(released_qc)), key=extract_number)

# Load scraped_qc
scraped_qc = []
for sheet in overview_sheets:
    df = pd.read_excel(excel_file_path, sheet_name=sheet, header=None)
    if len(df.columns) > 12:
        col_m = df.iloc[:, 12]
        col_n = df.iloc[:, 13]
        scraped_header = col_m.iloc[0]
        if scraped_header == "Scraped":
            col_m_values = col_m.iloc[3:].dropna().tolist()
            col_n_values = col_n.iloc[3:].dropna().tolist()
            combined_list = col_m_values + col_n_values
            used_columns = "M, N"
            if len(df.columns) > 14:
                third_row_o = df.iloc[2, 14]
                if third_row_o == "Anodes" or third_row_o == "Cathodes":
                    col_o = df.iloc[:, 14]
                    col_o_values = col_o.iloc[3:].dropna().tolist()
                    combined_list.extend(col_o_values)
                    used_columns += ", O"
            if len(df.columns) > 15:
                third_row_p = df.iloc[2, 15]
                if third_row_p == "Anodes" or third_row_p == "Cathodes":
                    col_p = df.iloc[:, 15]
                    col_p_values = col_p.iloc[3:].dropna().tolist()
                    combined_list.extend(col_p_values)
                    used_columns += ", P"
            scraped_qc.extend(combined_list)

scraped_qc_cleaned = sorted(list(set(scraped_qc)), key=extract_number)

# Extract B electrodes from metadata
B_electrodes = sorted(
    [item for item, desc in zip(metadata_df['Marking'].dropna(),
                                metadata_df.loc[metadata_df['Marking'].notna(), 'Description'].fillna(''))
     if str(item).startswith('B') and str(desc).strip().upper() in ['CATHODE', 'ANODE']],
    key=extract_number
)

# Extract electrode ranges
electrode_range_data = []
for sheet in overview_sheets:
    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheet, header=None)
        if len(df.columns) > 20:
            col_u = df.iloc[:, 20]
            col_v = df.iloc[:, 21]
            electrode_range_idx = None
            for idx, val in col_u.items():
                if pd.notna(val) and str(val).strip() == "Electrode Range":
                    electrode_range_idx = idx
                    break
            if electrode_range_idx is not None:
                start_row = electrode_range_idx + 1
                for row_idx in range(start_row, len(df)):
                    electrode_val = col_u.iloc[row_idx]
                    status_val = col_v.iloc[row_idx]
                    if pd.notna(electrode_val) and str(electrode_val).strip() and str(electrode_val).strip().startswith('B'):
                        electrode_range_data.append({
                            'Sheet': sheet,
                            'Electrode range': str(electrode_val).strip(),
                            'Status': str(status_val).strip() if pd.notna(status_val) else ''
                        })
                    elif pd.notna(electrode_val) and str(electrode_val).strip() and not str(electrode_val).strip().startswith('B'):
                        break
    except Exception as e:
        pass

electrode_status_ranges_df = pd.DataFrame(electrode_range_data)

# Filter B_electrodes based on range
if len(electrode_status_ranges_df) > 0:
    last_range = electrode_status_ranges_df.iloc[-1]['Electrode range']
    if '-' in str(last_range):
        end_electrode = str(last_range).split('-')[1].strip()
    else:
        end_electrode = str(last_range).strip()
    end_number = extract_number(end_electrode)
    B_electrodes = [item for item in B_electrodes if extract_number(item) <= end_number]

# Create all_tested and B_electrodes_extended
all_tested = sorted(list(set(released_qc_cleaned) | set(scraped_qc_cleaned)), key=extract_number)
tested_not_in_b = [item for item in all_tested if item not in B_electrodes]
B_electrodes_extended = B_electrodes + tested_not_in_b
B_electrodes_extended = sorted(B_electrodes_extended, key=extract_number)

# Create initial table_data
table_data = []
for electrode in B_electrodes_extended:
    in_released = electrode in released_qc_cleaned
    in_scraped = electrode in scraped_qc_cleaned
    is_tested = in_released or in_scraped
    released = in_released if is_tested else None
    scraped_val = in_scraped if is_tested else None
    table_data.append({
        'Electrode': electrode,
        'Tested': is_tested,
        'Released': released,
        'Scraped': scraped_val
    })

results_df_v1 = pd.DataFrame(table_data)

# Apply mark_scraped_neighbors
def mark_scraped_neighbors(df):
    for idx, row in df.iterrows():
        if row['Electrode'] == 'B2110' and row['Scraped'] == True:
            for offset in [-2, -1, 1, 2]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(df) and df.at[neighbor_idx, 'Released'] != True:
                    df.at[neighbor_idx, 'Scraped'] = True
    return df

results_df_v2 = mark_scraped_neighbors(results_df_v1)
electrode_analysis_df = results_df_v2.copy()

# Process electrode ranges
def parse_electrode_range(range_str, df):
    range_str = str(range_str).strip()
    if '-' in range_str:
        parts = range_str.split('-')
        start_electrode = parts[0].strip()
        end_electrode = parts[1].strip()
        start_num = extract_number(start_electrode)
        end_num = extract_number(end_electrode)
        df_electrodes = df['Electrode'].tolist()
        df_numbers = [extract_number(e) for e in df_electrodes]
        matching_indices = [i for i, num in enumerate(df_numbers) if start_num <= num <= end_num]
        if matching_indices:
            return [df_electrodes[i] for i in matching_indices]
        else:
            return []
    else:
        return [range_str]

if len(electrode_status_ranges_df) > 0:
    first_range = electrode_status_ranges_df.iloc[0]['Electrode range']
    if '-' in str(first_range):
        start_electrode = str(first_range).split('-')[0].strip()
    else:
        start_electrode = str(first_range).strip()
    start_number = extract_number(start_electrode)
    start_idx = electrode_analysis_df[electrode_analysis_df['Electrode'] == start_electrode].index.tolist()
    if start_idx:
        start_idx = start_idx[0]
        for idx in range(0, start_idx):
            released = electrode_analysis_df.at[idx, 'Released']
            scraped = electrode_analysis_df.at[idx, 'Scraped']
            if pd.isna(released) and pd.isna(scraped):
                electrode_analysis_df.at[idx, 'Released'] = True
                electrode_analysis_df.at[idx, 'Scraped'] = False
            elif released == True:
                electrode_analysis_df.at[idx, 'Scraped'] = False
            elif scraped == True:
                electrode_analysis_df.at[idx, 'Released'] = False

for idx, row in electrode_status_ranges_df.iterrows():
    electrode_range = row['Electrode range']
    status = row['Status'].strip()
    sheet = row['Sheet']
    electrodes_in_range = parse_electrode_range(electrode_range, electrode_analysis_df)
    updated_count = 0
    for electrode in electrodes_in_range:
        mask = electrode_analysis_df['Electrode'] == electrode
        if mask.any():
            if status == 'GOOD':
                electrode_analysis_df.loc[mask, 'Released'] = True
                electrode_analysis_df.loc[mask, 'Scraped'] = False
                updated_count += 1
            elif status == 'BAD (safety margin around scrap)' or status == 'NEEDS TESTING':
                electrode_analysis_df.loc[mask, 'Scraped'] = True
                electrode_analysis_df.loc[mask, 'Released'] = False
                updated_count += 1

# ============================================================================
# NOW PERFORM THE REQUESTED ANALYSIS
# ============================================================================

print("=" * 80)
print("ANALYSIS: BAD TESTED ELECTRODES AND THEIR GOOD NEIGHBORS")
print("=" * 80)

# Get tested electrodes (Tested=True)
tested_df = electrode_analysis_df[electrode_analysis_df['Tested'] == True].copy()

# Get bad tested electrodes (Tested=True and Scraped=True)
bad_tested = tested_df[tested_df['Scraped'] == True].copy()

# Get good tested electrodes (Tested=True and Released=True)
good_tested = tested_df[tested_df['Released'] == True].copy()

print(f"\n📊 SUMMARY STATISTICS")
print(f"  Total tested electrodes: {len(tested_df)}")
print(f"  Good tested electrodes (Released): {len(good_tested)}")
print(f"  Bad tested electrodes (Scraped): {len(bad_tested)}")
print(f"  Defect rate: {len(bad_tested)/len(tested_df)*100:.2f}%")

# Build analysis data for each bad electrode
analysis_results = []

for idx, bad_row in bad_tested.iterrows():
    bad_electrode = bad_row['Electrode']
    bad_electrode_num = extract_number(bad_electrode)
    
    # Find closest good tested before
    good_before = good_tested[good_tested.index < idx]
    closest_before = None
    closest_before_distance = None
    
    if len(good_before) > 0:
        closest_before_idx = good_before.index[-1]  # Last one before this index
        closest_before = good_before.iloc[-1]
        closest_before_distance = idx - closest_before_idx
    
    # Find closest good tested after
    good_after = good_tested[good_tested.index > idx]
    closest_after = None
    closest_after_distance = None
    
    if len(good_after) > 0:
        closest_after_idx = good_after.index[0]  # First one after this index
        closest_after = good_after.iloc[0]
        closest_after_distance = closest_after_idx - idx
    
    # Extract numeric values for distance calculation
    before_distance_rows = closest_before_distance if closest_before_distance is not None else np.nan
    after_distance_rows = closest_after_distance if closest_after_distance is not None else np.nan
    
    # Calculate numeric electrode distances
    before_distance_electrodes = None
    after_distance_electrodes = None
    
    if closest_before is not None:
        before_distance_electrodes = bad_electrode_num - extract_number(closest_before['Electrode'])
    
    if closest_after is not None:
        after_distance_electrodes = extract_number(closest_after['Electrode']) - bad_electrode_num
    
    analysis_results.append({
        'Bad Electrode': bad_electrode,
        'Bad Electrode Num': bad_electrode_num,
        'Good Before': closest_before['Electrode'] if closest_before is not None else 'N/A',
        'Good Before Num': extract_number(closest_before['Electrode']) if closest_before is not None else np.nan,
        'Distance Before (rows)': before_distance_rows,
        'Distance Before (electrodes)': before_distance_electrodes,
        'Good After': closest_after['Electrode'] if closest_after is not None else 'N/A',
        'Good After Num': extract_number(closest_after['Electrode']) if closest_after is not None else np.nan,
        'Distance After (rows)': after_distance_rows,
        'Distance After (electrodes)': after_distance_electrodes,
    })

analysis_df = pd.DataFrame(analysis_results)

print("\n" + "=" * 80)
print("DETAILED ANALYSIS - EACH BAD ELECTRODE AND ITS NEIGHBORS")
print("=" * 80)
print(analysis_df.to_string(index=False))

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICS ON BAD ELECTRODES AND NEIGHBORS")
print("=" * 80)

# Filter out NaN values for statistics
before_distances_rows = analysis_df['Distance Before (rows)'].dropna()
after_distances_rows = analysis_df['Distance After (rows)'].dropna()
before_distances_elec = analysis_df['Distance Before (electrodes)'].dropna()
after_distances_elec = analysis_df['Distance After (electrodes)'].dropna()

print(f"\n📏 DISTANCE METRICS (rows in dataframe)")
print(f"\nGood tested electrode BEFORE bad electrode:")
print(f"  Count with predecessor: {len(before_distances_rows)}/{len(analysis_df)}")
if len(before_distances_rows) > 0:
    print(f"  Mean distance: {before_distances_rows.mean():.2f} rows")
    print(f"  Median distance: {before_distances_rows.median():.2f} rows")
    print(f"  Std Dev: {before_distances_rows.std():.2f} rows")
    print(f"  Min distance: {before_distances_rows.min():.0f} rows")
    print(f"  Max distance: {before_distances_rows.max():.0f} rows")

print(f"\nGood tested electrode AFTER bad electrode:")
print(f"  Count with successor: {len(after_distances_rows)}/{len(analysis_df)}")
if len(after_distances_rows) > 0:
    print(f"  Mean distance: {after_distances_rows.mean():.2f} rows")
    print(f"  Median distance: {after_distances_rows.median():.2f} rows")
    print(f"  Std Dev: {after_distances_rows.std():.2f} rows")
    print(f"  Min distance: {after_distances_rows.min():.0f} rows")
    print(f"  Max distance: {after_distances_rows.max():.0f} rows")

print(f"\n📏 DISTANCE METRICS (electrode numbers)")
print(f"\nGood tested electrode BEFORE bad electrode:")
print(f"  Count with predecessor: {len(before_distances_elec)}/{len(analysis_df)}")
if len(before_distances_elec) > 0:
    print(f"  Mean distance: {before_distances_elec.mean():.2f} electrodes")
    print(f"  Median distance: {before_distances_elec.median():.2f} electrodes")
    print(f"  Std Dev: {before_distances_elec.std():.2f} electrodes")
    print(f"  Min distance: {before_distances_elec.min():.0f} electrodes")
    print(f"  Max distance: {before_distances_elec.max():.0f} electrodes")

print(f"\nGood tested electrode AFTER bad electrode:")
print(f"  Count with successor: {len(after_distances_elec)}/{len(analysis_df)}")
if len(after_distances_elec) > 0:
    print(f"  Mean distance: {after_distances_elec.mean():.2f} electrodes")
    print(f"  Median distance: {after_distances_elec.median():.2f} electrodes")
    print(f"  Std Dev: {after_distances_elec.std():.2f} electrodes")
    print(f"  Min distance: {after_distances_elec.min():.0f} electrodes")
    print(f"  Max distance: {after_distances_elec.max():.0f} electrodes")

# Isolation analysis
print(f"\n🔍 ISOLATION ANALYSIS")
both_neighbors = analysis_df[(analysis_df['Good Before'].notna()) & (analysis_df['Good Before'] != 'N/A') & 
                             (analysis_df['Good After'].notna()) & (analysis_df['Good After'] != 'N/A')]
no_before = analysis_df[(analysis_df['Good Before'].isna()) | (analysis_df['Good Before'] == 'N/A')]
no_after = analysis_df[(analysis_df['Good After'].isna()) | (analysis_df['Good After'] == 'N/A')]

print(f"  Bad electrodes with both good neighbors: {len(both_neighbors)}")
print(f"  Bad electrodes with NO good neighbor before: {len(no_before)}")
print(f"  Bad electrodes with NO good neighbor after: {len(no_after)}")

# Proximity analysis
if len(before_distances_rows) > 0 or len(after_distances_rows) > 0:
    # Count how many are "close" to good electrodes (within 10 rows)
    close_before = len(before_distances_rows[before_distances_rows <= 10])
    close_after = len(after_distances_rows[after_distances_rows <= 10])
    print(f"\n  Bad electrodes within 10 rows of previous good: {close_before}/{len(before_distances_rows)}")
    print(f"  Bad electrodes within 10 rows of next good: {close_after}/{len(after_distances_rows)}")

# Save results to CSV
analysis_df.to_csv("data/bad_electrode_neighbor_analysis.csv", index=False)
print(f"\n✅ Detailed analysis saved to: data/bad_electrode_neighbor_analysis.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distance before (rows)
if len(before_distances_rows) > 0:
    axes[0, 0].hist(before_distances_rows, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(before_distances_rows.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {before_distances_rows.mean():.1f}')
    axes[0, 0].set_xlabel('Distance (rows)', fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontweight='bold')
    axes[0, 0].set_title('Distance to Previous Good Electrode (rows)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

# Plot 2: Distance after (rows)
if len(after_distances_rows) > 0:
    axes[0, 1].hist(after_distances_rows, bins=20, color='#ff7f0e', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(after_distances_rows.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {after_distances_rows.mean():.1f}')
    axes[0, 1].set_xlabel('Distance (rows)', fontweight='bold')
    axes[0, 1].set_ylabel('Count', fontweight='bold')
    axes[0, 1].set_title('Distance to Next Good Electrode (rows)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

# Plot 3: Distance before (electrode numbers)
if len(before_distances_elec) > 0:
    axes[1, 0].hist(before_distances_elec, bins=20, color='#2ca02c', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(before_distances_elec.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {before_distances_elec.mean():.1f}')
    axes[1, 0].set_xlabel('Distance (electrode numbers)', fontweight='bold')
    axes[1, 0].set_ylabel('Count', fontweight='bold')
    axes[1, 0].set_title('Distance to Previous Good Electrode (electrode #s)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

# Plot 4: Distance after (electrode numbers)
if len(after_distances_elec) > 0:
    axes[1, 1].hist(after_distances_elec, bins=20, color='#d62728', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(after_distances_elec.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {after_distances_elec.mean():.1f}')
    axes[1, 1].set_xlabel('Distance (electrode numbers)', fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontweight='bold')
    axes[1, 1].set_title('Distance to Next Good Electrode (electrode #s)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/bad_electrode_neighbor_distances.png', dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: data/bad_electrode_neighbor_distances.png\n")
plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

