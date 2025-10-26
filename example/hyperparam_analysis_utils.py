import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import optuna
from matplotlib.lines import Line2D

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_trial_folder_name(trial_name, approach):
    if approach == "DSM2-Direct":
        return f"V2_Direct_{trial_name}_results"
    elif approach == "DSM2-Contrast":
        return f"V2_Contrastive_{trial_name}_results"
    elif approach == "NoDSM2-Direct":
        return f"NoDsm2_Direct_{trial_name}_results"
    elif approach == "NoDSM2-Contrast":
        return f"NoDsm2_Contrastive_{trial_name}_results"


def load_trial_station_csv(trial_name, approach, root="."):
    folder_name = get_trial_folder_name(trial_name, approach)
    full_path = os.path.join(root, folder_name)
    if not os.path.isdir(full_path):
        print(f"[load_trial_station_csv] => folder not found: {full_path}")
        return None

    csv_path = os.path.join(full_path, "trial_evaluation_metrics_amended.csv")
    if not os.path.exists(csv_path):
        print(f"[load_trial_station_csv] => CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[load_trial_station_csv] => CSV is empty => {csv_path}")
        return None

    return df


def gather_stations_with_all_hparams(df, trial_list, approach, scenario="base", root="."):
    rows = []
    master_df = df.get(approach)
    for tname in trial_list:
        if not isinstance(master_df, pd.DataFrame): continue
        if tname not in master_df.index: continue
        csv_path = os.path.join(root, get_trial_folder_name(tname, approach), "trial_evaluation_metrics_amended.csv")
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        if df.empty: continue
        try:
            row_master = master_df.loc[tname]
        except:
            row_master = None
        if "model" in df.columns:
            if scenario == "base":
                sub = df[df["model"] == "base.suisun-secondary"]
            else:
                sub = df[df["model"] == "base.suisun"]
            for _, r in sub.iterrows():
                st = r["station"]
                # if st not in ["sal","rsl"]: continue
                if "nse" not in r: continue
                out = {"trial": tname, "station": st, "nse": r["nse"], "approach": approach, "scenario": scenario}
                for c in ["freeze_bool", "arch_type", "contrast_weight",
                          "dsm2_init_lr", "dsm2_main_lr", "dsm2_init_epochs", "dsm2_main_epochs",
                          "schism_init_lr", "schism_main_lr", "schism_init_epochs", "schism_main_epochs",
                          "suisun_init_lr", "suisun_main_lr", "suisun_init_epochs", "suisun_main_epochs",
                          "base_init_lr", "base_main_lr", "base_init_epochs", "base_main_epochs"]:
                    if c in r:
                        out[c] = r[c]
                    elif row_master is not None and c in row_master:
                        out[c] = row_master[c]
                    else:
                        out[c] = np.nan
                rows.append(out)
        else:
            col_nse = f"{scenario}_nse"
            for _, r in df.iterrows():
                if col_nse not in r: continue
                out = {"trial": tname, "station": r["station"], "nse": r[col_nse], "approach": approach,
                       "scenario": scenario}
                for c in ["freeze_bool", "arch_type", "contrast_weight",
                          "dsm2_init_lr", "dsm2_main_lr", "dsm2_init_epochs", "dsm2_main_epochs",
                          "schism_init_lr", "schism_main_lr", "schism_init_epochs", "schism_main_epochs",
                          "suisun_init_lr", "suisun_main_lr", "suisun_init_epochs", "suisun_main_epochs",
                          "base_init_lr", "base_main_lr", "base_init_epochs", "base_main_epochs"]:
                    if c in r:
                        out[c] = r[c]
                    elif row_master is not None and c in row_master:
                        out[c] = row_master[c]
                    else:
                        out[c] = np.nan
                rows.append(out)
    return pd.DataFrame(rows)


def plot_box_hparam(df, hyper_col, y_col, approach_order=None, hue_order=None, title=None, figsize=(10, 6),
                    rotation=20):
    if approach_order is None:
        approach_order = ["DSM2-Direct", "DSM2-Contrast", "NoDSM2-Direct", "NoDSM2-Contrast"]

    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x="approach", y=y_col, hue=hyper_col, order=approach_order, hue_order=hue_order)
    plt.title(title if title else f"{y_col} by approach (hue: {hyper_col})")
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_box_base_suisun(df, approach_name, group_col, base_col="mean_nse_base", suisun_col="mean_nse_suisun",
                         title=None, figsize=(6, 4), rotation=20):
    sub_df = df[df["approach"] == approach_name].copy()
    sub_melted = sub_df.melt(id_vars=[group_col], value_vars=[base_col, suisun_col],
                             var_name="scenario", value_name="nse")

    plt.figure(figsize=figsize)
    sns.boxplot(data=sub_melted, x="scenario", y="nse", hue=group_col)
    plt.title(f"{approach_name} => {base_col} & {suisun_col} by {group_col}" if not title else title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_box_all_combos(df, approach_name, group_cols, base_col="mean_nse_base", suisun_col="mean_nse_suisun",
                        combo_order=None, title=None, figsize=(10, 5), rotation=20):
    sub_df = df[df["approach"] == approach_name].copy()
    needed_cols = group_cols + [base_col, suisun_col]
    sub_df = sub_df[needed_cols].copy()
    melted = sub_df.melt(id_vars=group_cols, value_vars=[base_col, suisun_col], var_name="scenario", value_name="nse")
    melted = melted.reset_index(drop=True)

    plt.figure(figsize=figsize)
    sns.boxplot(data=melted, x=melted.apply(lambda row: "_".join(str(row[gc]) for gc in group_cols), axis=1), y="nse",
                hue="scenario", order=combo_order)
    plt.xticks(rotation=rotation)

    if not title:
        title = f"{approach_name} => {base_col} & {suisun_col} by {group_cols}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_approach_by_station(df, station, hyper_col=None, approach_order=None, hue_order=None, title=None, height=5,
                             aspect=1.2, rotation=20):
    if approach_order is None:
        approach_order = ["DSM2-Direct", "NoDSM2-Direct", "DSM2-Contrast", "NoDSM2-Contrast"]

    scenario_order = ["base", "suisun"]

    sub_df = (df[(df["station"] == station) & (df["scenario"].isin(scenario_order))].copy())

    plot_kwargs = dict(data=sub_df, x="approach", y="nse", col="scenario", kind="box", order=approach_order,
                       sharey=False, height=height, aspect=aspect, col_order=scenario_order)

    if hyper_col:
        plot_kwargs["hue"] = hyper_col
        if hue_order:
            plot_kwargs["hue_order"] = hue_order

    g = sns.catplot(**plot_kwargs)
    g.set_xticklabels(rotation=rotation)
    g.set_axis_labels("Approach", "NSE")
    g.set_titles("Scenario = {col_name}")

    if hyper_col:
        g._legend.set_title(hyper_col)
    else:
        g._legend.remove()

    if title is None:
        title = f"NSE @ {station} by approach & scenario"
        if hyper_col:
            title += f", hue={hyper_col}"
    g.fig.suptitle(title, y=1.03, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_scenario_box_for_station(df, approach_name, station_name, hyper_col, scenario_order=("base", "suisun"),
                                  hue_order=None, title=None, figsize=(6, 4), rotation=20):
    sub_df = df[(df["approach"] == approach_name) & (df["station"] == station_name) &
                (df["scenario"].isin(scenario_order))].copy()

    plt.figure(figsize=figsize)
    sns.boxplot(data=sub_df, x="scenario", y="nse", hue=hyper_col, order=scenario_order, hue_order=hue_order)
    plt.xticks(rotation=rotation)

    if not title:
        title = f"{approach_name} => {station_name.upper()} NSE by scenario, hue={hyper_col}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_station_hparam_scenario_x(df, approach_name, station_name, scenario_list=("base", "suisun"),
                                   group_cols=("freeze_bool", "arch_type"), scenario_order=None, combo_order=None,
                                   title=None, figsize=(8, 4), rotation=20):
    if scenario_order is None:
        scenario_order = scenario_list
    sub = df[(df["approach"] == approach_name) & (df["station"] == station_name) &
             (df["scenario"].isin(scenario_list))].copy()

    if sub.empty:
        print(f"[WARN] No data for approach={approach_name}, station={station_name}, scenarios={scenario_list}")
        return

    sub = sub.reset_index(drop=True)

    for gc in group_cols:
        sub[gc] = sub[gc].astype(str)
    if len(group_cols) == 0:
        sub["combo"] = "All"
    else:
        sub["combo"] = sub.apply(lambda row: "_".join(str(row[gc]) for gc in group_cols), axis=1)

    plt.figure(figsize=figsize)
    sns.boxplot(data=sub, x="scenario", y="nse", hue="combo", order=scenario_order, hue_order=combo_order)
    plt.xticks(rotation=rotation)

    if not title:
        title = f"{approach_name} - station={station_name}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def prepare_station_nse_wide(df, station_name="sal"):
    df_station = df[df["station"] == station_name].copy()
    id_cols = [c for c in df_station.columns if c not in ("station", "scenario", "nse")]
    df_base = df_station[df_station["scenario"] == "base"].copy()
    df_suisun = df_station[df_station["scenario"] == "suisun"].copy()
    df_base.drop(columns=["station", "scenario"], inplace=True)
    df_suisun.drop(columns=["station", "scenario"], inplace=True)
    df_base.rename(columns={"nse": f"{station_name}_base_nse"}, inplace=True)
    df_suisun.rename(columns={"nse": f"{station_name}_suisun_nse"}, inplace=True)
    merged = df_base.merge(df_suisun, on=id_cols, how="inner")
    merged[f"avg_{station_name}_nse"] = (merged[f"{station_name}_base_nse"] + merged[f"{station_name}_suisun_nse"]) / 2
    return merged


def plot_station_nse_across_approaches(df, base_col="sal_base_nse", suisun_col="sal_suisun_nse", title=None,
                                       figsize=(8, 5), rotation=20):
    melted = df.melt(id_vars=["approach"], value_vars=[base_col, suisun_col], var_name="scenario", value_name="nse")
    plt.figure(figsize=figsize)
    sns.boxplot(data=melted, x="scenario", y="nse", hue="approach")
    if not title:
        title = "Base vs Suisun NSE by Approach"
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_two_scenario_boxplots_approach_x_hparam_hue(df, hue_col, approach_col="approach", base_col="sal_base_nse",
                                                     suisun_col="sal_suisun_nse", suptitle=None, figsize=(10, 4),
                                                     rotation=20):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=False)
    df[hue_col] = df[hue_col].astype(str)

    sns.boxplot(data=df, x=approach_col, y=base_col, hue=hue_col, ax=axes[0])
    axes[0].set_title("Base NSE")
    axes[0].tick_params(axis="x", rotation=rotation)

    sns.boxplot(data=df, x=approach_col, y=suisun_col, hue=hue_col, ax=axes[1])
    axes[1].set_title("Suisun NSE")
    axes[1].tick_params(axis="x", rotation=rotation)
    axes[0].legend_.remove()

    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.03)

    plt.tight_layout()
    plt.show()


def plot_box_scenario_station(df, approach_name, base_col, suisun_col, group_col=None, title=None, figsize=(6, 4),
                              rotation=20):
    sub_df = df[df["approach"] == approach_name].copy()

    if group_col:
        sub_df[group_col] = sub_df[group_col].astype(str)
        sub_melted = sub_df.melt(id_vars=[group_col], value_vars=[base_col, suisun_col], var_name="scenario",
                                 value_name="nse")
    else:
        sub_melted = sub_df.melt(value_vars=[base_col, suisun_col], var_name="scenario", value_name="nse")

    plt.figure(figsize=figsize)
    if group_col:
        sns.boxplot(data=sub_melted, x="scenario", y="nse", hue=group_col)
    else:
        sns.boxplot(data=sub_melted, x="scenario", y="nse")

    if not title:
        title = f"{approach_name} => Base & Suisun NSE" + (f" by {group_col}" if group_col else "")
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_box_scenario_combos(df, approach_name, base_col, suisun_col, group_cols=None, combo_order=None, title=None,
                             figsize=(8, 4), rotation=20):
    if group_cols is None:
        group_cols = []
    sub_df = df[df["approach"] == approach_name].copy()

    if len(group_cols) == 0:
        sub_df["combo"] = "All"
    else:
        for gc in group_cols:
            sub_df[gc] = sub_df[gc].astype(str)
        sub_df["combo"] = sub_df.apply(lambda row: "_".join(str(row[gc]) for gc in group_cols), axis=1)

    sub_melted = sub_df.melt(id_vars=["combo"], value_vars=[base_col, suisun_col], var_name="scenario",
                             value_name="nse")

    plt.figure(figsize=figsize)
    sns.boxplot(data=sub_melted, x="combo", y="nse", hue="scenario", order=combo_order)

    if not title:
        if len(group_cols) == 0:
            title = f"{approach_name} => Base & Suisun NSE (no grouping)"
        else:
            title = f"{approach_name} => Base & Suisun NSE by {group_cols}"
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def prepare_surrogate_data(df, hyperparams, target_col):
    needed_cols = hyperparams + [target_col]
    sub = df[needed_cols].copy()
    sub = sub.dropna(subset=[target_col])
    y = sub[target_col].values
    X = sub.drop(columns=[target_col])
    if "freeze_bool" in X.columns:
        X["freeze_bool"] = X["freeze_bool"].astype(int)

    if "arch_type" in X.columns:
        arch_dummies = pd.get_dummies(X["arch_type"], prefix="arch")
        X = pd.concat([X.drop(columns=["arch_type"]), arch_dummies], axis=1)

    feature_names = list(X.columns)
    return X, y, feature_names


def train_random_forest(X, y, test_size=0.2, random_state=42, n_estimators=100, max_depth=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    metrics = {"r2_train": r2_score(y_train, y_pred_train), "r2_test": r2_score(y_test, y_pred_test),
               "mse_train": mean_squared_error(y_train, y_pred_train),
               "mse_test": mean_squared_error(y_test, y_pred_test)}
    return model, metrics, X_train, X_test, y_train, y_test


def train_xgboost(X, y, test_size=0.2, random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                             random_state=random_state)

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {"r2_train": r2_score(y_train, y_pred_train), "r2_test": r2_score(y_test, y_pred_test),
               "mse_train": mean_squared_error(y_train, y_pred_train),
               "mse_test": mean_squared_error(y_test, y_pred_test)}
    return model, metrics, X_train, X_test, y_train, y_test


def print_feature_importances(model, feature_names, top_n=20):
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("Feature Importances (descending):")
    for i, (f, imp) in enumerate(feat_imp[:top_n], start=1):
        print(f"{i:2d}) {f:<25} importance={imp:.4f}")


def train_rf_optuna(X, y, n_trials=30, random_state=42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    def objective_rf(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      random_state=random_state, n_jobs=-1)

        cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring="neg_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_rf, n_trials=n_trials)
    best_params = study.best_params

    final_model = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
    final_model.fit(X_trainval, y_trainval)

    y_pred_test = final_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    result = {"test_mse": mse_test, "test_r2": r2_test, "best_params": best_params}
    return final_model, result, (X_trainval, X_test, y_trainval, y_test)


def train_xgboost_optuna(X, y, n_trials=30, random_state=42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    def objective_xgb(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 12)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.5, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)

        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                 subsample=subsample, colsample_bytree=colsample_bytree, random_state=random_state)

        cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring="neg_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_xgb, n_trials=n_trials)
    best_params = study.best_params

    final_model = xgb.XGBRegressor(**best_params, random_state=random_state)
    final_model.fit(X_trainval, y_trainval)

    y_pred_test = final_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    result = {"test_mse": mse_test, "test_r2": r2_test, "best_params": best_params}

    return final_model, result, (X_trainval, X_test, y_trainval, y_test)


def pick_top_trials_any_approach(df_all: pd.DataFrame, metric_col: str, N: int,
                                 is_min_better: bool = False) -> pd.DataFrame:
    """
    Return a dataframe with only the best N rows across all approaches
    according to `metric_col`.
    """
    df_sorted = df_all.sort_values(metric_col, ascending=is_min_better)
    return df_sorted.head(N).reset_index(drop=True)


def gather_station_df_general(df_trials_subset: pd.DataFrame, metric: str, scenario: str = "base",
                              root: str = ".") -> pd.DataFrame:
    """
    Build the wide trial×station dataframe for ANY mixture of approaches.
    Expects columns: trial_name, approach  in df_trials_subset.
    """
    rows = []

    for _, rec in df_trials_subset.iterrows():
        trial_name = rec["trial_name"]
        approach = rec["approach"]

        if approach == "DSM2-Direct":
            folder = f"V2_Direct_{trial_name}_results"
            suffix = "direct"
        elif approach == "DSM2-Contrast":
            folder = f"V2_Contrastive_{trial_name}_results"
            suffix = "contrast"
        elif approach == "NoDSM2-Direct":
            folder = f"NoDsm2_Direct_{trial_name}_results"
            suffix = "direct"
        elif approach == "NoDSM2-Contrast":
            folder = f"NoDsm2_Contrastive_{trial_name}_results"
            suffix = "contrast"
        else:
            continue

        csv_path = os.path.join(root, folder, "trial_evaluation_metrics.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] missing {csv_path}")
            continue
        df_m = pd.read_csv(csv_path)
        if df_m.empty:
            continue

        row = {"trial": trial_name, "approach": approach}

        if suffix == "direct":
            colname = f"{scenario}_{metric}"
            for _, r in df_m.iterrows():
                row[r["station"]] = r[colname]
        else:
            model_tag = "base.suisun-secondary" if scenario == "base" else "base.suisun"
            sub = df_m[df_m["model"] == model_tag]
            for _, r in sub.iterrows():
                row[r["station"]] = r[metric]

        rows.append(row)

    df_wide = pd.DataFrame(rows).fillna(np.nan)
    return df_wide


def parallel_plot_stations(df_plot: pd.DataFrame, title: str, is_min_better: bool, style_map=None,
                           figsize: tuple = (14, 6)):
    if style_map is None:
        style_map = {"DSM2-Direct": dict(color="red", ls="-", abbrev="D"),
                     "DSM2-Contrast": dict(color="orange", ls="--", abbrev="C"),
                     "NoDSM2-Direct": dict(color="blue", ls=":", abbrev="ND"),
                     "NoDSM2-Contrast": dict(color="darkgreen", ls="-.", abbrev="NC")}

    need = {"trial", "approach"}

    if need.difference(df_plot.columns):
        raise ValueError("df_plot must contain 'trial' and 'approach' columns")

    station_cols = [c for c in df_plot.columns if c not in ("trial", "approach")]
    xvals = np.arange(len(station_cols))
    fig, ax = plt.subplots(figsize=figsize)

    for rank, (_, row) in enumerate(df_plot.iterrows(), start=1):
        sty = style_map.get(row["approach"], dict(color="grey", ls="-", abbrev="?"))
        line_label = f"{rank}-{sty['abbrev']}"
        yvals = [row[s] for s in station_cols]

        ax.plot(xvals, yvals, color=sty["color"], linestyle=sty["ls"], linewidth=2, label=line_label)

    for j, sc in enumerate(station_cols):
        vals = df_plot[sc]
        if vals.notna().sum() == 0:
            continue
        idx_best = vals.idxmin() if is_min_better else vals.idxmax()
        best_row = df_plot.loc[idx_best]
        sty_best = style_map.get(best_row["approach"], dict(color="black"))
        ax.scatter(j, best_row[sc], s=160, marker="o", edgecolor="k", zorder=5, color=sty_best["color"])

    ax.set_xticks(xvals)
    ax.set_xticklabels(station_cols, rotation=30, ha="right")
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    handles_trials, labels_trials = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles_trials, labels_trials, title="Top‑N trials  (rank‑abbr)", bbox_to_anchor=(1.02, 1.0),
                     loc="upper left", frameon=False)
    ax.add_artist(leg1)

    counts = df_plot["approach"].value_counts().to_dict()
    handles_app, labels_app = [], []

    for app, sty in style_map.items():
        if app not in counts:
            continue
        h = Line2D([0], [0], color=sty["color"], linestyle=sty["ls"], linewidth=3)
        handles_app.append(h)
        labels_app.append(f"{app}  ({counts[app]})")

    ax.legend(handles_app, labels_app, title="Approach  (hits in Top‑N)", bbox_to_anchor=(1.02, 0.45), loc="upper left",
              frameon=False)

    plt.tight_layout()
    plt.show()


def plot_topN(df_source, N, metric_rank_col, metric_station, scenario, title_prefix):
    """Pick Top‑N, convert to wide station frame, make the plot."""
    df_best = pick_top_trials_any_approach(df_source, metric_col=metric_rank_col, N=N, is_min_better=False)
    df_wide = gather_station_df_general(df_trials_subset=df_best, metric=metric_station, scenario=scenario)
    parallel_plot_stations(df_wide, title=f"{title_prefix} – {scenario.capitalize()} NSE", is_min_better=False)


def compute_metrics(y_true, y_pred):
    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))

    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "nse": np.nan, "pearson_r": np.nan}

    yt, yp = y_true[mask], y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    nse = 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    r = np.corrcoef(yt, yp)[0, 1] if len(yt) > 1 else np.nan

    return {"mae": mae, "rmse": rmse, "nse": nse, "pearson_r": r}


def load_and_merge(prefix, trial_suffix, ref_suffix="xvalid_ref_out_unscaled", ann_suffix="xvalid", ann_idx=0,
                   DATA_DIR="."):
    """Returns merged DF with *_pred columns, or None if missing."""
    ref_csv = f"{prefix}_{trial_suffix}_{ref_suffix}.csv"
    ann_csv = f"{prefix}_{trial_suffix}_{ann_suffix}_{ann_idx}.csv"
    ref_path, ann_path = os.path.join(DATA_DIR, ref_csv), os.path.join(DATA_DIR, ann_csv)

    if not (os.path.exists(ref_path) and os.path.exists(ann_path)):
        print("   missing:", ref_csv, ann_csv);
        return None

    df_ref = pd.read_csv(ref_path, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_path, parse_dates=["datetime"])

    return pd.merge(df_ref, df_ann, on=["datetime", "case"], how="inner",
                    suffixes=("", "_pred"))


def plot_timeseries(df, station, out_png, n_cases=7, title=None):
    st, st_pred = station, f"{station}_pred"

    if st not in df.columns or st_pred not in df.columns: return
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(n_cases, 1, figsize=(8, 2.4 * n_cases), constrained_layout=True)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes, start=1):
        sub = df[df["case"] == i]
        ax.plot(sub["datetime"], sub[st], color="0.1", label="Ref")
        ax.plot(sub["datetime"], sub[st_pred], label="ANN")
        ax.set_ylabel("Norm EC")
        ax.set_title(f"Case {i}  (#rows={len(sub)})")
        if i == 1 and title: ax.set_title(title + "\n" + ax.get_title())

    axes[0].legend()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
