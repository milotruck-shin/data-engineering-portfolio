# alarm_correlation.py

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import datetime as dt

PAIR_WINDOW_MINUTES = 5  # pairing window for OPTO3~OPTO6 proximity and outage matching
MIN_DURATION_MINUTES = 1
MAX_DURATION_HOURS = 24


def parse_args():
    parser = argparse.ArgumentParser(
        description="Alarm correlation script"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input CSV file or folder"
    )
    return parser.parse_args()


# **read_csv_kwargs**: Accept extra keyword arguments for pandas.read_csv
def load_csv_folder(folder: Path, pattern: str = "*.csv", **read_csv_kwargs) -> pd.DataFrame:
    csv_files = sorted(folder.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' in {folder}")
    dfs = []
    for f in csv_files:
        data_frame = pd.read_csv(f, **read_csv_kwargs)
        dfs.append(data_frame)
    return pd.concat(dfs, ignore_index=True)


def preprocess(
    df: pd.DataFrame,
    outage_alarms: list[str],
    alarm_order: list[str],
    min_duration_minutes: int = MIN_DURATION_MINUTES,
    max_duration_hours: int = MAX_DURATION_HOURS,
) -> pd.DataFrame:

    # Normalize outage alarm names
    df['Alarm Name'] = np.where(df['Alarm Name'].isin(outage_alarms), 'Outage Alarm', df['Alarm Name'])

    # Parse time columns
    df['First Occurred On'] = pd.to_datetime(df['First Occurred On'])
    df['Cleared On'] = pd.to_datetime(df['Cleared On'])
    df['Duration(hh:mm:ss)'] = pd.to_timedelta(df['Duration(hh:mm:ss)'])

    # Duration filtering
    df = df[
        (df['Duration(hh:mm:ss)'] >= pd.Timedelta(minutes=min_duration_minutes)) &
        (df['Duration(hh:mm:ss)'] < pd.Timedelta(hours=max_duration_hours))
    ].copy()

    # Drop unwanted column if present
    if 'Alarm Occurrences' in df.columns:
        df.drop(['Alarm Occurrences'], axis=1, inplace=True)

    # Add calendar date (date component of occurrence)
    Date = df['First Occurred On'].dt.date
    df.insert(2, 'Date', Date)

    # Categorical ordering for Alarm Name (only the alarms we care about)
    df["Alarm Name"] = (
        df["Alarm Name"].astype("category").cat.set_categories(alarm_order, ordered=True)
    )

    # Sort for deterministic processing
    df = df.sort_values(
        by=["Site Name", "Date", "Alarm Name", "First Occurred On"],
        ascending=[True, True, True, True]
    )

    # Convert Alarm Name to string for mapping
    df['Alarm Name'] = df['Alarm Name'].astype('string')

    # Map to Array: OPTO3->1, OPTO6->2, Outage->3
    array_map = {
        "OPTO 3 Main Failure": 1,
        "OPTO 6 Rectifier Failure": 2,
        "Outage Alarm": 3
    }
    df["Array"] = df["Alarm Name"].map(array_map)

    # Keep groups with >1 unique Array (for pairing), BUT also keep all Outage rows
    unique_counts = df.groupby(['Site Name', 'Date'])['Array'].transform('nunique')
    df = df[(unique_counts > 1) | (df['Alarm Name'] == 'Outage Alarm')].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def group_pairing(g: pd.DataFrame, used_out_global=None, pair_window_min: int = PAIR_WINDOW_MINUTES):
    """
    Pairing within an expanded per-day window (g contains core day + next-day Outage candidates).
    `used_out_global` prevents reusing the same Outage for the same site across adjacent days.
    """
    # Index arrays
    idx_3 = g.index[g["Array"] == 1].to_numpy()
    idx_6 = g.index[g["Array"] == 2].to_numpy()
    idx_out = g.index[g["Array"] == 3].to_numpy()

    # Containers
    groupA, groupB, groupC, groupD = [], [], [], []
    used_3, used_6, used_out = set(), set(), set()
    if used_out_global is None:
        used_out_global = set()

    # Extract times (occurrence and clearance)
    t3 = g.loc[idx_3, "First Occurred On"].values if len(idx_3) else np.array([])
    t6 = g.loc[idx_6, "First Occurred On"].values if len(idx_6) else np.array([])
    tOut_Occur = g.loc[idx_out, "First Occurred On"].values if len(idx_out) else np.array([])
    tOut_Clr = g.loc[idx_out, "Cleared On"].values if len(idx_out) else np.array([])
    t3Clr = g.loc[idx_3, "Cleared On"].values if len(idx_3) else np.array([])
    t6Clr = g.loc[idx_6, "Cleared On"].values if len(idx_6) else np.array([])

    # --------------------------------------------------
    # PHASE 1: OPTO3–OPTO6 pairing (Group A & B)
    # --------------------------------------------------
    if len(idx_3) > 0 and len(idx_6) > 0:
        # OPTO3–OPTO6 occurrence proximity (minutes)
        delta_min = np.abs(t3[:, None] - t6[None, :]) / np.timedelta64(1, "m")
        i, j = np.where(delta_min <= pair_window_min)
        # Sort candidate pairs by closeness (ascending time diff)
        pairs = sorted(zip(i, j), key=lambda x: delta_min[x[0], x[1]])

        for i3, i6 in pairs:
            if i3 in used_3 or i6 in used_6:
                continue  # enforce one-to-one pairing

            # Try Outage match first (Group A): clearance-times proximity
            if len(idx_out) > 0:
                diff3 = np.abs(tOut_Clr - t3Clr[i3]) / np.timedelta64(1, "m")
                diff6 = np.abs(tOut_Clr - t6Clr[i6]) / np.timedelta64(1, "m")
                valid = np.where((diff3 <= pair_window_min) & (diff6 <= pair_window_min))[0]
            else:
                valid = []

            matched = False
            for k in valid:
                out_idx = idx_out[k]
                if (k in used_out) or (out_idx in used_out_global):
                    continue
                # ensure Outage occurs after OPTO6 start
                if tOut_Occur[k] < t6[i6]:
                    continue

                used_out.add(k)
                used_out_global.add(out_idx)
                used_3.add(i3)
                used_6.add(i6)

                groupA.append({
                    "idx_3": idx_3[i3],
                    "idx_6": idx_6[i6],
                    "idx_out": out_idx
                })
                matched = True
                break  # take the closest valid outage only

            # If no Outage matched -> Group B
            if not matched:
                diff = np.abs(t6Clr[i6] - t3Clr[i3]) / np.timedelta64(1, "m")
                if diff <= pair_window_min:
                    if i3 in used_3 or i6 in used_6:
                        continue
                    used_3.add(i3)
                    used_6.add(i6)
                    groupB.append({
                        "idx_3": idx_3[i3],
                        "idx_6": idx_6[i6]
                    })

    # --------------------------------------------------
    # PHASE 2: leftover OPTO3 (Group C) - only OPTO3 with Outage
    # --------------------------------------------------
    if len(idx_3) > 0 and len(idx_out) > 0 and len(idx_6) == 0:
        for i3 in range(len(idx_3)):
            diff = np.abs(tOut_Clr - t3Clr[i3]) / np.timedelta64(1, "m")
            valid = np.where(diff <= pair_window_min)[0]
            for k in valid:
                out_idx = idx_out[k]
                if (k in used_out) or (out_idx in used_out_global):
                    continue
                if tOut_Occur[k] < t3[i3]:
                    continue
                used_out.add(k)
                used_out_global.add(out_idx)
                groupC.append({
                    "idx_3": idx_3[i3],
                    "idx_out": out_idx
                })
                break

    # --------------------------------------------------
    # PHASE 3: leftover OPTO6 (Group D) - only OPTO6 with Outage
    # --------------------------------------------------
    if len(idx_6) > 0 and len(idx_out) > 0 and len(idx_3) == 0:
        for i6 in range(len(idx_6)):
            diff = np.abs(tOut_Clr - t6Clr[i6]) / np.timedelta64(1, "m")
            valid = np.where(diff <= pair_window_min)[0]
            for k in valid:
                out_idx = idx_out[k]
                if (k in used_out) or (out_idx in used_out_global):
                    continue
                if tOut_Occur[k] < t6[i6]:
                    continue
                used_out.add(k)
                used_out_global.add(out_idx)
                groupD.append({
                    "idx_6": idx_6[i6],
                    "idx_out": out_idx
                })
                break

    return groupA, groupB, groupC, groupD


def match_all_sites(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Site:
      - Iterate by Date.
      - Build a core frame of that Date's alarms (OPTO3/6 + any Outage on that day).
      - Expand Outage candidates to also include next day's Outage rows.
      - Run pairing with a site-level used-outage set to avoid double matches.
    """
    all_matches = []

    # Process per site to avoid cross-site mixing and to maintain a site-level used-outage set
    for site, site_df in df.groupby("Site Name"):
        used_out_global = set()  # avoid reusing an Outage across adjacent days for the same site
        dates = sorted(site_df["Date"].unique())

        for d in dates:
            # Core: alarms that occurred on day d
            g_core = site_df[site_df["Date"] == d]

            # Outage pool: outages from day d and day d+1
            d_next = d + dt.timedelta(days=1)
            outage_pool = site_df[
                (site_df["Alarm Name"] == "Outage Alarm") &
                (site_df["Date"].isin([d, d_next]))
            ]

            # Avoid duplicating rows already in core
            outage_pool = outage_pool.loc[~outage_pool.index.isin(g_core.index)]

            # Combine for pairing window
            g = pd.concat([g_core, outage_pool], axis=0)

            # Pair within this window; pass the site-level used-outage tracker
            groupA, groupB, groupC, groupD = group_pairing(
                g,
                used_out_global=used_out_global,
                pair_window_min=PAIR_WINDOW_MINUTES
            )
            all_matches.extend(groupA)
            all_matches.extend(groupB)
            all_matches.extend(groupC)
            all_matches.extend(groupD)

    if not all_matches:
        print("No matches found.")
        exit()

    match_df = pd.DataFrame(all_matches)

    # Choose anchor index per row: idx_3 when present, otherwise idx_6
    site_idx = match_df["idx_3"].combine_first(match_df["idx_6"])

    # Build result dataframe
    res_df = pd.DataFrame({
        "Site Name":        df["Site Name"].reindex(site_idx).to_numpy(),
        "Date":             df["Date"].reindex(site_idx).to_numpy(),

        "OPTO3_occur":      df["First Occurred On"].reindex(match_df["idx_3"]).to_numpy(),
        "OPTO6_occur":      df["First Occurred On"].reindex(match_df["idx_6"]).to_numpy(),
        "Outage_occur":     df["First Occurred On"].reindex(match_df["idx_out"]).to_numpy(),

        "OPTO3_clearance":  df["Cleared On"].reindex(match_df["idx_3"]).to_numpy(),
        "OPTO6_clearance":  df["Cleared On"].reindex(match_df["idx_6"]).to_numpy(),
        "Outage_clearance": df["Cleared On"].reindex(match_df["idx_out"]).to_numpy(),
    })

    # Assign group labels A/B/C/D
    res_df["Group"] = np.select(
        [
            res_df["OPTO3_clearance"].notna() &
            res_df["OPTO6_clearance"].notna() &
            res_df["Outage_clearance"].notna(),

            res_df["OPTO3_clearance"].notna() &
            res_df["OPTO6_clearance"].notna() &
            res_df["Outage_clearance"].isna(),

            res_df["OPTO3_clearance"].notna() &
            res_df["OPTO6_clearance"].isna() &
            res_df["Outage_clearance"].notna(),

            res_df["OPTO3_clearance"].isna() &
            res_df["OPTO6_clearance"].notna() &
            res_df["Outage_clearance"].notna(),
        ],
        ["A", "B", "C", "D"],
        default=pd.NA
    )

    return res_df


def calculate_dur(res_df: pd.DataFrame):
    # Initialize columns
    res_df["Backup Duration"] = np.nan
    res_df["Power Outage"] = np.nan
    res_df["Site Outage"] = np.nan

    mask_c = res_df["Group"] == "C"
    mask_d = res_df["Group"] == "D"
    mask_a = res_df["Group"] == "A"

    # ----Group C----#
    res_df.loc[mask_c, "Backup Duration"] = (
        res_df.loc[mask_c, "Outage_occur"] - res_df.loc[mask_c, "OPTO3_occur"]
    ).dt.total_seconds() / 3600
    res_df.loc[mask_c, "Power Outage"] = (
        res_df.loc[mask_c, "Outage_clearance"] - res_df.loc[mask_c, "OPTO3_occur"]
    ).dt.total_seconds() / 3600

    # ----Group D----#
    res_df.loc[mask_d, "Backup Duration"] = (
        res_df.loc[mask_d, "Outage_occur"] - res_df.loc[mask_d, "OPTO6_occur"]
    ).dt.total_seconds() / 3600
    res_df.loc[mask_d, "Power Outage"] = (
        res_df.loc[mask_d, "Outage_clearance"] - res_df.loc[mask_d, "OPTO6_occur"]
    ).dt.total_seconds() / 3600

    # ----Group A----#
    res_df.loc[mask_a, "Backup Duration"] = (
        res_df.loc[mask_a, "Outage_occur"] - res_df.loc[mask_a, "OPTO6_occur"]
    ).dt.total_seconds() / 3600
    res_df.loc[mask_a, "Power Outage"] = (
        res_df.loc[mask_a, "Outage_clearance"] - res_df.loc[mask_a, "OPTO6_occur"]
    ).dt.total_seconds() / 3600

    # Site Outage = outage cleared - outage occurred
    mask = res_df["Outage_occur"].notna() & res_df["Outage_clearance"].notna()
    res_df.loc[mask, "Site Outage"] = (
        res_df.loc[mask, "Outage_clearance"] - res_df.loc[mask, "Outage_occur"]
    ).dt.total_seconds() / 3600

    cols = ["Backup Duration", "Power Outage", "Site Outage"]
    res_df[cols] = res_df[cols].round(3)

    return res_df


######################################################################################################


if __name__ == "__main__":
    # Categorize Outage Alarms using vectorized operations
    outage_alarms = ['CSL Fault', 'OML Fault', 'S1ap Link Down_NE Down', 'epsEnodebunreachable_NE Down']
    alarm_order = ["OPTO 3 Main Failure", "OPTO 6 Rectifier Failure", "Outage Alarm"]

    args = parse_args()
    input_path = args.input_path

    if input_path.is_dir():
        out_path = input_path.joinpath("result.csv")
        if out_path.exists():
            out_path.unlink()   # delete old result.csv
        df = load_csv_folder(input_path, pattern="*.csv")
    elif input_path.is_file():
        df = pd.read_csv(input_path)
        out_path = input_path.with_name("result.csv")
    else:
        raise ValueError("Input path must be a file or directory")

    df = preprocess(df, outage_alarms, alarm_order)
    res_df = match_all_sites(df)
    res_df = calculate_dur(res_df)

    # SAVE RESULT
    res_df.to_csv(out_path, index=False)

    print(res_df.head())
    print("\nGroup counts:")
    print(res_df["Group"].value_counts(dropna=False))
