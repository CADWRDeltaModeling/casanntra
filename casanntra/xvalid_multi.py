from casanntra.model_builder import *
import concurrent.futures
import matplotlib.pyplot as plt
from casanntra.single_or_list import *
import traceback
import pickle
import base64
import pandas as pd


def _multi_scenario_tags(builder, count):
    is_ms = getattr(builder, "is_multi_scenario_step", lambda: False)()
    if not is_ms:
        return [str(i) for i in range(count)]

    try:
        sc_cfg = builder.builder_args.get("scenarios", []) or []
    except Exception:
        sc_cfg = []

    tags = ["base"] + [sc.get("id", f"scenario{i}") for i, sc in enumerate(sc_cfg, start=1)]
    if len(tags) != count:
        tags = ["base"] + [f"scenario{i}" for i in range(1, count)]
    return tags


def _ordered_outputs(builder, outputs):
    tags = _multi_scenario_tags(builder, len(outputs))
    return list(zip(outputs, tags))


def single_model_fit(
    builder,
    df_in,
    fit_in,
    fit_out,
    test_in,
    test_out,
    out_prefix,
    init_train_rate,
    init_epochs,
    main_train_rate,
    main_epochs):

    input_layers = builder.input_layers()
    ann = builder.build_model(input_layers, df_in)

    print("Fitting model in single_model_fit")
    try:
        print("Fit input keys:", list(fit_in.keys()))
    except Exception:
        print("Fit input type:", type(fit_in))

    try:
        exp_inputs = [t.name for t in ann.inputs]
    except Exception:
        exp_inputs = ann.inputs

    print("Model expected inputs:", exp_inputs)
    print("Should be the same")

    history, ann = builder.fit_model(
        ann,
        fit_in,
        fit_out,
        test_in,
        test_out,
        init_train_rate=init_train_rate,
        init_epochs=init_epochs,
        main_train_rate=main_train_rate,
        main_epochs=main_epochs)

    print("Predicting data in single_model_fit")
    test_pred = ann.predict(test_in)
    print("Prediction complete")
    del ann
    print(f"Return type {type(test_pred)}")
    return test_pred


def bulk_fit(
    builder,
    df_in,
    df_out,
    out_prefix,
    fit_in,
    fit_out,
    test_in,
    test_out,
    init_train_rate,
    init_epochs,
    main_train_rate,
    main_epochs,
):
    """Uses the ingredients of xvalid_multi but does a single fit with all the data
    for situations like exporting the model where a single version of the model is needed.
    """

    # ✅ Ensure df_in and df_out are correctly formatted
    df_in = df_in.copy()
    df_in["ifold"] = 0

    if isinstance(df_out, list):
        df_out = [x.copy() for x in df_out]
        for df in df_out:
            df["ifold"] = 0

    # ✅ Apply antecedent preservation to aligned inputs
    # There will be only one set of inputs to the ANN (no source and target like outputs)
    inputs_lagged = builder.calc_antecedent_preserve_cases(df_in)
    outputs_trim = (
        [df.loc[inputs_lagged.index, builder.output_list()] for df in df_out]
        if isinstance(df_out, list)
        else df_out.loc[inputs_lagged.index, builder.output_list()]
    )

    fit_in = inputs_lagged
    fit_out = outputs_trim
    test_in = fit_in
    test_out = outputs_trim

    # ✅ Convert DataFrame inputs into structured dicts for multi-input models
    idx = pd.IndexSlice
    fit_in = builder.df_by_feature_and_time(fit_in).drop(
        ["datetime", "case", "fold"], level="var", axis=1
    )
    fit_in = {
        name: fit_in.loc[:, idx[name, :]].droplevel("var", axis=1)
        for name in builder.input_names
    }
    test_in = builder.df_by_feature_and_time(test_in).drop(
        ["datetime", "case", "fold"], level="var", axis=1
    )
    test_in = {
        name: test_in.loc[:, idx[name, :]].droplevel("var", axis=1)
        for name in builder.input_names
    }

    input_layers = builder.input_layers()
    ann = builder.build_model(input_layers, df_in)

    history, ann = builder.fit_model(
        ann,
        fit_in,
        fit_out,
        test_in,
        test_out,
        init_train_rate=init_train_rate,
        init_epochs=init_epochs,
        main_train_rate=main_train_rate,
        main_epochs=main_epochs,
    )

    test_pred = ann.predict(test_in)

    return ann


def reorder(pkeys):
    """Reorder the keys to match the order of the output_list"""
    pkeys = sorted(pkeys, key=lambda x: (("target" not in x), ("source" not in x), x))
    return pkeys


def xvalid_fit_multi(
    df_in,
    df_out,
    builder,
    init_train_rate,
    init_epochs,
    main_train_rate,
    main_epochs,
    out_prefix,
    pool_size):
    """
    Splits input by fold, fits models per fold (withheld CV), merges predictions,
    and writes x-valid CSVs. Multi-scenario outputs are written with descriptive,
    scenario-id-based filenames.
    """

    num_outputs = builder.num_outputs()          
    output_list = builder.output_list()          

    inputs_lagged = builder.calc_antecedent_preserve_cases(df_in)

    if isinstance(df_out, list):
        outputs_trim = [dfo.loc[inputs_lagged.index, :] for dfo in df_out]
    else:
        outputs_trim = df_out.loc[inputs_lagged.index, :]

    outputs_xvalid, histories = allocate_receiving_df(outputs_trim, output_list), {}

    futures = []
    foldmap = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:
        for ifold in df_in.fold.unique():
            print(f"Scheduling fit for fold {ifold}")

            fit_in = inputs_lagged.loc[inputs_lagged.fold != ifold, :]
            test_in = inputs_lagged.loc[inputs_lagged.fold == ifold, :]

            if isinstance(df_out, list):
                fit_out = [df.loc[fit_in.index, output_list] for df in df_out]
                test_out = [df.loc[test_in.index, output_list] for df in df_out]
                rep = fit_out[0]
                print(f"ifold={ifold} # train input rows={fit_in.shape[0]} # train out rows={rep.shape[0]}")
                print(f"ifold={ifold} # test input rows={test_in.shape[0]} # test  out rows={rep.shape[0]}")
            else:
                fit_out = df_out.loc[fit_in.index, output_list]
                test_out = df_out.loc[test_in.index, output_list]
                print(f"ifold={ifold} # train input rows={fit_in.shape[0]} # train out rows={fit_out.shape[0]}")
                print(f"ifold={ifold} # test input rows={test_in.shape[0]} # test  out rows={test_out.shape[0]}")

            idx = pd.IndexSlice
            fit_in_split = builder.df_by_feature_and_time(fit_in).drop(["datetime", "case", "fold"], level="var", axis=1)
            fit_in_split = {name: fit_in_split.loc[:, idx[name, :]].droplevel("var", axis=1) for name in builder.input_names}

            test_in_split = builder.df_by_feature_and_time(test_in).drop(["datetime", "case", "fold"], level="var", axis=1)
            test_in_split = {name: test_in_split.loc[:, idx[name, :]].droplevel("var", axis=1) for name in builder.input_names}

            future = executor.submit(
                single_model_fit,
                builder,
                df_in,
                fit_in_split,
                fit_out,
                test_in_split,
                test_out,
                out_prefix=out_prefix,
                init_epochs=init_epochs,
                init_train_rate=init_train_rate,
                main_epochs=main_epochs,
                main_train_rate=main_train_rate)
            
            futures.append(future)
            foldmap[future] = ifold

    for future in concurrent.futures.as_completed(futures):
        try:
            ifold = foldmap[future]
            test_pred = future.result()
            print(f"Test prediction data type: {type(test_pred)}")
            test_in = inputs_lagged.loc[inputs_lagged.fold == ifold, :]

            if isinstance(outputs_xvalid, list):
                print("\nUpdating master xvalidation structure (multi-output version)")

                if isinstance(test_pred, dict):
                    pkeys = list(test_pred.keys())
                    map_fn = getattr(builder, "map_prediction_keys_to_outputs", None)
                    if callable(map_fn):
                        supervised_keys = map_fn(pkeys)
                    else:
                        supervised_keys = None

                    if supervised_keys:
                        if len(supervised_keys) != len(outputs_xvalid):
                            raise ValueError(
                                f"Head mapping mismatch: {len(supervised_keys)} keys vs {len(outputs_xvalid)} receivers"
                            )
                        for j, key in enumerate(supervised_keys):
                            outputs_xvalid[j].loc[test_in.index, output_list] = test_pred[key]
                    else:
                        ordered = reorder(pkeys)
                        j = 0
                        for key in ordered:
                            if "contrast" in key:
                                continue
                            outputs_xvalid[j].loc[test_in.index, output_list] = test_pred[key]
                            j += 1
                else:
                    for j in range(len(outputs_xvalid)):
                        outputs_xvalid[j].loc[test_in.index, output_list] = test_pred[j]

            elif isinstance(outputs_xvalid, pd.DataFrame):
                if isinstance(test_pred, dict):
                    if len(test_pred) != 1:
                        raise ValueError("Multiple outputs returned for a single-output model.")
                    test_pred = list(test_pred.values())[0]
                print("\nUpdating master xvalidation structure (single-output version)")
                outputs_xvalid.loc[test_in.index, output_list] = test_pred

            else:
                raise ValueError("Unsupported outputs_xvalid container type.")

        except Exception as err:
            print(f"Exception likely in fold {ifold}")
            traceback.print_tb(err.__traceback__)
            raise err

    print("Writing master xvalidation outputs to file(s)")

    if isinstance(outputs_xvalid, list):
        ordered = _ordered_outputs(builder, outputs_xvalid)
        for df_recv, tag in ordered:
            outxfile = f"{out_prefix}_xvalid_{tag}.csv"
            df_recv[output_list] = df_recv[output_list].astype(float)
            df_recv.to_csv(
                outxfile,
                float_format="%.3f",
                date_format="%Y-%m-%dT%H:%M",
                header=True,
                index=True,
            )
            print(f"[OK] wrote {outxfile}")

    else:
        outputs_xvalid[output_list] = outputs_xvalid[output_list].astype(float)
        outxfile = f"{out_prefix}_xvalid.csv"
        outputs_xvalid.to_csv(
            outxfile,
            float_format="%.3f",
            date_format="%Y-%m-%dT%H:%M",
            header=True,
            index=True,
        )
        print(f"[OK] wrote {outxfile}")

    print("Done writing\n")
    return outputs_xvalid, histories


@single_or_list("df_out")
def allocate_receiving_df(df_out, column_list):
    # ✅ Pre-allocate multiple datastructures for holding results based on num_outputs
    output_cols = ["datetime", "case", "fold"] + column_list
    outputs_xvalid = pd.DataFrame(columns=output_cols, index=df_out.index)
    # ✅ Preserve datetime, case, and fold info
    outputs_xvalid.loc[:, ["datetime", "case", "fold"]] = df_out.loc[
        :, ["datetime", "case", "fold"]
    ]
    return outputs_xvalid
