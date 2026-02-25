from pathlib import Path
import argparse
import pandas as pd
import numpy as np

PAIR_WINDOW_MINUTES = 1  #pairing window for OPTO3~OPTO6 and outage matching
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

#**read_csv_kwargs : “Accept any extra keyword arguments that were not explicitly listed in the function definition, 
# and collect them into a dictionary called read_csv_kwargs.”

def load_csv_folder(folder : Path, pattern: str="*.csv",**read_csv_kwargs) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError (f"No CSV files matching '{pattern} in {folder}") #raise ExceptionType("message")
    dfs=[]

    for f in csv_files:
        data_frame = pd.read_csv(f,**read_csv_kwargs)
        dfs.append(data_frame)
    return (pd.concat(dfs, ignore_index=True))

def preprocess(df:pd.DataFrame,outage_alarms:list[str],alarm_order:list[str],
    min_duration_minutes: int = MIN_DURATION_MINUTES,
    max_duration_hours: int = MAX_DURATION_HOURS,
    ) -> pd.DataFrame:

    df['Alarm Name'] = np.where(df['Alarm Name'].isin(outage_alarms), 'Outage Alarm', df['Alarm Name'])
    df['First Occurred On'] = pd.to_datetime(df['First Occurred On'])
    df['Cleared On'] = pd.to_datetime(df['Cleared On'])
    df['Duration(hh:mm:ss)'] = pd.to_timedelta(df['Duration(hh:mm:ss)'])
    df = df[
        (df['Duration(hh:mm:ss)'] > pd.Timedelta(minutes=min_duration_minutes)) &
        (df['Duration(hh:mm:ss)'] < pd.Timedelta(hours=max_duration_hours))
    ]

    #to delete any unwanted columns
    delete_col=['Alarm Occurrences']
    df.drop(delete_col, axis=1, inplace=True)


    mask = df["Alarm Name"] == 'Outage Alarm'
    df_outage = df.loc[mask].copy()
    df_opto = df.loc[~mask].copy()
    del df

    Date = df_opto['First Occurred On'].dt.date
    cleared_opto = df_opto["Cleared On"].dt.date
    df_opto.insert(2, 'Date', Date)
    df_opto.insert(3, 'Cleared_date', cleared_opto)


    cat_type = pd.CategoricalDtype(categories=alarm_order, ordered=True)
    df_opto['Alarm Name'] = df_opto['Alarm Name'].astype(cat_type)
    df_opto = df_opto.sort_values(
        by=["Site Name", "Date","Alarm Name","First Occurred On"],
        ascending=[True, True, True, True]
    )

    df_opto['Alarm Name'] = df_opto['Alarm Name'].astype('string')   # or .astype('object')

    #create a column called Array. Correalation: OPTO 3 -> 1, OPTO 6 -> 2, Outage -> 3
    array_map = {
        "OPTO 3 Main Failure": 1,
        "OPTO 6 Rectifier Failure": 2,
        "Outage Alarm": 3
    }

    df_opto["Array"] = df_opto["Alarm Name"].map(array_map)
    #df_outage["Array"] = df_outage["Alarm Name"].map(array_map)

    df_opto.reset_index(drop=True, inplace=True)  #removes the old index (which now has gaps after filtering). Creates a clean 0, 1, 2, ... index
    df_outage.reset_index(drop=True, inplace=True)
    return df_opto, df_outage

def group_pairing(g):
    #index arrays
    g = g.reset_index(drop=True)
    arr=g["Array"].to_numpy()
    idx_3   = np.where(arr == 1)[0]  #og df index
    idx_6   = np.where(arr==2)[0]

    #all the extracted values here are from both OPTO3 and OPTO6. the idx_x array will differentiate which they belong to
    t_occ = g["First Occurred On"].to_numpy("datetime64[ns]")
    t_clr = g["Cleared On"].to_numpy("datetime64[ns]")
    site_arr = g["Site Name"].to_numpy(object)  
    date_arr = g["Date"].to_numpy(object)
    cleared_opto_date = g["Cleared_date"].to_numpy(object)

    #allocate some containers to  indicate which opto alarms have been used for matching 
    used_3 = np.zeros(len(idx_3), dtype = bool)  #0 for unused, 1 for used. 
    used_6 = np.zeros(len(idx_6), dtype = bool)  

    #extract times (occurrence and clearance)
    t3_occ = t_occ[idx_3]      #the index of this t3 is not same as og index. same size as idx_3 or idx_6
    t6_occ = t_occ[idx_6]

    t3_clr = t_clr[idx_3]      #the index of this t3 is not same as og index. same size as idx_3 or idx_6
    t6_clr = t_clr[idx_6]
    
    rows = []
    # --------------------------------------------------
    # PHASE 1: OPTO3–OPTO6 pairing (Group A & B)
    # --------------------------------------------------
    if len(idx_3) > 0 and len(idx_6) > 0:

      #broadcast subtraction. final output is nxm dimension. N for opto3 times, M for opto 6 times.
      #produces a boolean matrix of the shape (NxM), then produces indices that satisfy the condition.
      #i[k], j[k] is the position of the k-th valid pairing
      #sort candidate pairs by closeness (ascending time diff)

        diff_occ = np.abs(t3_occ[:, None] - t6_occ[None, :]) / np.timedelta64(1, "m") 
        diff_clr = np.abs(t3_clr[:, None] - t6_clr[None, :]) / np.timedelta64(1, "m") 
        valid_pairs = (
            (diff_occ <= PAIR_WINDOW_MINUTES) &
            (diff_clr <=PAIR_WINDOW_MINUTES)
        )

        i, j = np.where(valid_pairs)


        pairs = sorted(zip(i, j), key=lambda x: diff_occ[x[0], x[1]])              #zip object returns ((i0,j0), (i1,j1),(i2,j2), ...) each tuple is a OPTO3-OPTO6 match
                                                                                    #sorting key, for each pair, look up the time difference
        for i3, i6 in pairs:

            if used_3[i3] or used_6[i6]:
                continue   # enforce one-to-one pairing.

            used_3[i3] = True
            used_6[i6] = True

            #map the matrix of time delta<5 back to og dataframe index
            opto3 = idx_3[i3]
            opto6 = idx_6[i6]
            
            site_name = site_arr[opto3]
            date = date_arr[opto3]
            cleared_date = cleared_opto_date[opto3]

            o3_occ = t_occ[opto3]
            o6_occ = t_occ[opto6]
            o3_clr = t_clr[opto3]
            o6_clr = t_clr[opto6]


            rows.append({
                "Site Name":site_name,
                "Date":date,
                "Cleared Date":cleared_date,
                "OPTO3_occur":o3_occ,
                "OPTO6_occur":o6_occ,
                "Outage_occur":pd.NaT,
                "OPTO3_clearance":o3_clr,
                "OPTO6_clearance":o6_clr,
                "Outage_clearance":pd.NaT,
            })

        #mark the leftover opto3 and opto6 alarms
        leftover_idx3 = idx_3[~used_3]
        leftover_idx6 = idx_6[~used_6]

        if len(leftover_idx3)>0:
            for x in leftover_idx3:

                site_name = site_arr[x]
                date = date_arr[x]
                cleared_date = cleared_opto_date[x]
                o3_occ = t_occ[x]
                o3_clr = t_clr[x]

                rows.append({
                "Site Name":site_name,
                "Date":date,
                "Cleared Date":cleared_date,
                "OPTO3_occur":o3_occ,
                "OPTO6_occur":pd.NaT,
                "Outage_occur":pd.NaT,
                "OPTO3_clearance":o3_clr,
                "OPTO6_clearance":pd.NaT,
                "Outage_clearance":pd.NaT,
            })
                
        if len(leftover_idx6)>0:
            for x in leftover_idx6:

                site_name = site_arr[x]
                date = date_arr[x]
                cleared_date = cleared_opto_date[x]
                o6_occ = t_occ[x]
                o6_clr = t_clr[x]

                rows.append({
                "Site Name":site_name,
                "Date":date,
                "Cleared Date":cleared_date,
                "OPTO3_occur":pd.NaT,
                "OPTO6_occur":o6_occ,
                "Outage_occur":pd.NaT,
                "OPTO3_clearance":pd.NaT,
                "OPTO6_clearance":o6_clr,
                "Outage_clearance":pd.NaT,
            })


    return rows

def single_alarm_group(g):
    g = g.reset_index(drop=True)

    df = pd.DataFrame({
        "Site Name": g["Site Name"].values,
        "Date": g["Date"].values,
        "Cleared Date": g["Cleared_date"].values,
        "OPTO3_occur": pd.NaT,
        "OPTO6_occur": pd.NaT,
        "Outage_occur": pd.NaT,
        "OPTO3_clearance": pd.NaT,
        "OPTO6_clearance": pd.NaT,
        "Outage_clearance": pd.NaT
    })

    alarm_type = g["Array"].iloc[0]

    if alarm_type==1: #OPTO 3
        df["OPTO3_occur"]=g["First Occurred On"].values
        df["OPTO3_clearance"]=g["Cleared On"].values

    else: #OPTO6
        df["OPTO6_occur"]=g["First Occurred On"].values
        df["OPTO6_clearance"]=g["Cleared On"].values

    return df.to_dict('records')

def normalise_outage_format(df: pd.DataFrame) -> pd.DataFrame:

    Date = df["First Occurred On"].dt.date
    Cleared_date = df["Cleared On"].dt.date
    df.insert(2, 'Date', Date)
    df.insert(3, 'Cleared Date', Cleared_date)
    #Cleared date is used to catergorise outages with OPTO alarms later.

    return pd.DataFrame({
        "Site Name":df["Site Name"],
        "Date":df["Date"],
        "Cleared Date":df["Cleared Date"],
        "OPTO3_occur": pd.NaT,
        "OPTO6_occur": pd.NaT,
        "Outage_occur":df["First Occurred On"],
        "OPTO3_clearance": pd.NaT,
        "OPTO6_clearance": pd.NaT,
        "Outage_clearance":df["Cleared On"],
    })

def concatenate_dfs(df: pd.DataFrame, df_outage: pd.DataFrame) -> pd.DataFrame:
    all_matches = []


    for _, g in df.groupby(["Site Name", "Date"]):

        if g["Array"].nunique()>1:
            sort1 = group_pairing(g)
            all_matches.extend(sort1)
        else:
            sort2 = single_alarm_group(g)
            all_matches.extend(sort2)

    df_total = pd.DataFrame(all_matches)

    #combine both DataFrames 
    final_df = pd.concat([df_total, df_outage], ignore_index=True)
    final_df["Cleared Date"] = pd.to_datetime(final_df["Cleared Date"])
    final_df["Site Name"] = final_df["Site Name"].astype(str)

    final_df = final_df.sort_values(by=["Site Name", "Cleared Date"],ascending=[True, True])

    return final_df

def match_all_sites(final_df: pd.DataFrame) -> pd.DataFrame:
    #Classify the final rows baed on which row columns data at OPTO3,6 and Outages.
    #to be used as pruning, or further processing

    #create boolean masks (1 if value exists, 0 if NaN value)
    has_opto3 = final_df["OPTO3_occur"].notna()
    has_opto6 = final_df["OPTO6_occur"].notna()
    has_outage = final_df["Outage_occur"].notna()

    conditions = [(has_opto3 & has_opto6),(has_opto3 ^ has_opto6),has_outage]
    catergory = [2,1,3]
    final_df["Array"] = np.select(conditions,catergory,default=0)
    output=[]

    for _,g in final_df.groupby(["Site Name","Cleared Date"]):
        if (g["Array"].nunique() == 1):    #group only has opto3 or only opto6 alarms. skip this

            if (g["Array"].iloc[0] == 1):
                continue
            elif (g["Array"].iloc[0] == 2):
                output.extend(g.to_dict('records'))
                continue

        elif (g["Array"].nunique() >1):  #could be (opto3, opto6), (opto3&6), outage
            g.reset_index(drop=True, inplace=True)
            opto_idx = g.index[g["Array"].isin([1,2])]
            outage_idx = g.index[g["Array"]==3]

            if len(outage_idx)==0:      #if the groups only have opto3/6 and opto3-6 pairs, take the opto3-6 pairs only and put in a list dict. this is group B
                B_rows = g.loc[g["Array"]==2].to_dict('records')
                output.extend(B_rows)
                continue

            else:

                # For OPTOs: Use OPTO6 clearance, if missing use OPTO3 clearance
                """
                t_opto_clear = g.loc[opto_idx, "OPTO6_clearance"].combine_first(g.loc[opto_idx, "OPTO3_clearance"]).values
                t_out_clear = g.loc[outage_idx, "Outage_clearance"].values

                t_opto_occur = g.loc[opto_idx, "OPTO6_occur"].combine_first(g.loc[opto_idx, "OPTO3_occur"]).values
                t_out_occur = g.loc[outage_idx, "Outage_occur"].values
                """

                t_o3_clr = g.loc[opto_idx, "OPTO3_clearance"].values.astype("datetime64[ns]")
                t_o6_clr = g.loc[opto_idx, "OPTO6_clearance"].values.astype("datetime64[ns]")
                t_out_clr = g.loc[outage_idx, "Outage_clearance"].values.astype("datetime64[ns]")

                t_o3_occ = g.loc[opto_idx, "OPTO3_occur"].values.astype("datetime64[ns]")
                t_o6_occ = g.loc[opto_idx, "OPTO6_occur"].values.astype("datetime64[ns]")
                t_out_occ = g.loc[outage_idx, "Outage_occur"].values.astype("datetime64[ns]")


                # STEP A: OCCURRENCE CHECK (Outage >= First OPTO Alarm)
                diff_occ_3 = t_out_occ[None, :] - t_o3_occ[:,None]      #matrix : outage occ - opto3 occur
                diff_occ_6 = t_out_occ[None, :] - t_o6_occ[:,None]      #matrix : outage occ - opto6 occur


                # logic: valid if (Diff >= 0) OR (OPTO is Missing/NaT)
                # treat NaT comparisons as False, so we explicitly allow NaT
                valid_occ3 = (diff_occ_3 >= pd.Timedelta(0)) | np.isnat(t_o3_occ[:, None])
                valid_occ6 = (diff_occ_6 >= pd.Timedelta(0)) | np.isnat(t_o6_occ[:, None])
                
                # MUST be valid against all existing alarms in the row
                valid_occ = valid_occ3 & valid_occ6

                # STEP B: CLEARANCE CHECK (Strict 5 Min Window)
                diff_clr3 = np.abs(t_out_clr[None, :] - t_o3_clr[:, None])   # Matrix 1: Outage vs OPTO3
                diff_clr6 = np.abs(t_out_clr[None, :] - t_o6_clr[:, None])   # Matrix 2: Outage vs OPTO6

                
                # Logic: (Diff <= 5) OR (OPTO is Missing)
                # If OPTO3 is NaT (missing), the check passes automatically for that column
                valid_clr3 = (diff_clr3 <= pd.Timedelta(minutes=PAIR_WINDOW_MINUTES)) | np.isnat(t_o3_clr[:, None])
                valid_clr6 = (diff_clr6 <= pd.Timedelta(minutes=PAIR_WINDOW_MINUTES)) | np.isnat(t_o6_clr[:, None])
                
                # COMBINE: The Outage must satisfy BOTH existing alarms
                valid_clr = valid_clr3 & valid_clr6
                
                valid_match = valid_clr & valid_occ
                i,j=np.where(valid_match)

                d3_fill = diff_clr3.copy()
                d3_fill[np.isnat(d3_fill)] = pd.Timedelta(0)
                
                d6_fill = diff_clr6.copy()
                d6_fill[np.isnat(d6_fill)] = pd.Timedelta(0)
                
                # Get max difference for every pair (e.g. max(4min, 8min) = 8min)
                max_diffs = np.maximum(d3_fill, d6_fill)
                
                # Sort based on this max difference
                match_diffs = max_diffs[i, j]
                sorted_indices = np.argsort(match_diffs)
                i = i[sorted_indices]
                j = j[sorted_indices]

                used_optos = set()
                used_outages = set()

                for ii in range(len(i)):
                    opto_local_idx = opto_idx[i[ii]]
                    outage_local_idx = outage_idx[j[ii]]

                    if opto_local_idx in used_optos or outage_local_idx in used_outages:
                        continue

                    g.at[opto_local_idx, "Outage_occur"] = g.at[outage_local_idx, "Outage_occur"]
                    g.at[opto_local_idx, "Outage_clearance"] = g.at[outage_local_idx, "Outage_clearance"]
            
                    used_optos.add(opto_local_idx)
                    used_outages.add(outage_local_idx)

                results = g.loc[opto_idx].copy()
                output.extend(results.to_dict('records'))

                all_outage_indices = set(outage_idx)
                unused_outages = list(all_outage_indices - used_outages)
                
                if unused_outages:
                    results_outage = g.loc[unused_outages].copy()
                    output.extend(results_outage.to_dict('records'))
    
    return pd.DataFrame(output)

def group_labelling(df: pd.DataFrame) ->pd.DataFrame:
    df["Group"] = np.select(
        [
            df["OPTO3_clearance"].notna() &
            df["OPTO6_clearance"].notna() &
            df["Outage_clearance"].notna(),

            df["OPTO3_clearance"].notna() &
            df["OPTO6_clearance"].notna() &
            df["Outage_clearance"].isna(),

            df["OPTO3_clearance"].notna() &
            df["OPTO6_clearance"].isna() &
            df["Outage_clearance"].notna(),

            df["OPTO3_clearance"].isna() &
            df["OPTO6_clearance"].notna() &
            df["Outage_clearance"].notna(),
        ],
        ["A", "B", "C", "D"],
        default=pd.NA
    )

    df = df.dropna(subset=["Group"])
    df = df.drop(columns=["Array"])

    return df

def calculate_dur(df : pd.DataFrame):
    #for duration calculation, check what group and write cases. for exp, group C, only OPTO3 and Outage. and then do subtraction.
    
    #initialize columns (prevents leftovers)
    df["Backup Duration"] = np.nan
    df["Power Outage"] = np.nan
    df["Site Outage"] = np.nan

    mask_c = df["Group"] == "C"
    mask_d = df["Group"] == "D"
    mask_a = df["Group"] == "A"

    #----Group C----#
    df.loc[mask_c,"Backup Duration"] =(df.loc[mask_c,"Outage_occur"] - df.loc[mask_c,"OPTO3_occur"]).dt.total_seconds()/3600
    df.loc[mask_c,"Power Outage"] =(df.loc[mask_c,"Outage_clearance"] - df.loc[mask_c,"OPTO3_occur"]).dt.total_seconds()/3600

    #----Group D----#
    df.loc[mask_d,"Backup Duration"] =(df.loc[mask_d,"Outage_occur"] - df.loc[mask_d,"OPTO6_occur"]).dt.total_seconds()/3600
    df.loc[mask_d,"Power Outage"] =(df.loc[mask_d,"Outage_clearance"] - df.loc[mask_d,"OPTO6_occur"]).dt.total_seconds()/3600

    #----Group A----#
    df.loc[mask_a,"Backup Duration"] =(df.loc[mask_a,"Outage_occur"] - df.loc[mask_a,"OPTO6_occur"]).dt.total_seconds()/3600
    df.loc[mask_a,"Power Outage"] = (df.loc[mask_a,"Outage_clearance"] - df.loc[mask_a,"OPTO6_occur"]).dt.total_seconds()/3600

    #site outage = outage cleared - outage occured
    mask = df["Outage_occur"].notna() & df["Outage_clearance"].notna()
    df.loc[mask,"Site Outage"] =(df.loc[mask,"Outage_clearance"] - df.loc[mask,"Outage_occur"]).dt.total_seconds() / 3600

    cols = ["Backup Duration", "Power Outage", "Site Outage"]
    df[cols] = df[cols].round(3)

    return df


######################################################################################################

if __name__=="__main__":

    #categorize Alarms using np.where (vectorised operation), pls use vector operations (numpy) whenever possible
    outage_alarms = ['CSL Fault', 'OML Fault', 'S1ap Link Down_NE Down', 'epsEnodebunreachable_NE Down']
    alarm_order = ["OPTO 3 Main Failure", "OPTO 6 Rectifier Failure", "Outage Alarm"]

    args = parse_args()
    input_path = args.input_path
    if input_path.is_dir():

        out_path = input_path.joinpath("result.csv")

        if out_path.exists():
            out_path.unlink()   # delete old result.csv

        df = load_csv_folder(input_path)

    elif input_path.is_file():
        df = pd.read_csv(input_path)
        out_path = input_path.with_name("result.csv")

    else:
        raise ValueError("Input path must be a file or directory")
    
    df_opto, df_outage = preprocess(df,outage_alarms,alarm_order)
    df_outage = normalise_outage_format(df_outage)
    concat_df = concatenate_dfs(df_opto, df_outage)
    final_df = match_all_sites(concat_df)
    final_df = group_labelling(final_df)
    final_df=calculate_dur(final_df)

    # SAVE RESULT
    final_df.to_csv(out_path,index=False)


    print(final_df.head())
    print("\nGroup counts:")
    print(final_df["Group"].value_counts(dropna=False))
