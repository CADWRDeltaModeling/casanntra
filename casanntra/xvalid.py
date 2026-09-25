from casanntra.model_builder import *
import matplotlib.pyplot as plt


"""Typical workflow:

1. Read the raw data in. The data should have a "case" column. The columns datetime and case form a unique index. See read_data.read_data()

2. Perform any desired pre-model transformations where features become different features. Note that if you are 
   doing something like g() you may need to do it with something like:
   df['g'] = df.groupby("case").transform(["ndoi": my_gfunc])  (and make sure to use positivity preserving integration)
   ModelBuilder: 1. This is model dependent. Having transformations that are not part of TensorFlow is a consistency problem with Java, 
                 so avoid if you can. Do this outside the model
                 2. Need to initialize input_names and output_names to whatever the final names are that will be used by model. 
                 3. Need to make sure feature_dim will return the expected time dim for each feature (scalar) not including batch. For instance, if there are
                    90 lagged days for LSTM/GRU it will return 90 

3. Create lagged data (e.g. 7 individual lags and 10 11-day averaged blocks for classic CalSIM or 
   90 lagged individual for LSTM/GRU). This calcution needs to be done without crossing over cases. 
   See calc_antecedent_preserve_cases().
   ModelBuilder: Usually you can just follow the way the examples define the variables ndays, nwindows and window_length
4. Create inputs layers for the model. This is the first TensorFlow layer that receives the input and possibly tacks on
   some normalization. The product of this step is TensorFlow architecture, not data. However the input data 
   are passed in to allow scaling to be calculated. The created layers must have one named input 
   for each conceptual feature ("sac_flow"). The 
   lags will be part of the dimensionality. The first action within the ANN may concatenate these (or you might transform a 
   subset of the variables using linear layers and then concatenate). The desired dimension is (batch=None, ntime, nfeature).
   This has caused problems in published papers.
   ModelBuilder: You may want to over-ride this with tailored scaling or eliminate scaling altogether if you have a 
                 a transformation in mind (e.g. summing Sac, SJR, exports, CU to become a surrogate for outflow)
                 and you want to defer scaling until after that so that it is farther in the model.

5. Pre-calculate the cross-validation folds for the inputs. Originally this was done by leave-1-case out but the function
   xvalid_folds() also can split cases  with a target length like 180d (actual splits will be the same length within the case
   and will be at least the target length) which keeps more data available for training. 
   The identifier of the fold will be appended to the 'fold' column. 
   ModelBuilder: As long as you have the right columns (datetime,case) and a target size for the fold (e.g. 180d) the examples should be fine.

6. Extract the output columns. Need to discuss if there is a need/ability to scale these.
   ModelBuilder: If you have set the names of columns this should work. It reindexes the output to match the input.

7. Implement build_model (for architecture) and fit_model (for individual model fit). Every fold creates a new model, which is in anticipation of 
   multithreading.   
   ModelBuilder: build_model is where you define your achitecture after the input layer. fit_model is something you build as well
                 that describes  your fitting process.

7. Pass the inputs and outputs to the xvalid(input,output) member. This function will:
   a. iterate the folds, witholding (lagged) input and aligning output for each fold forming training and test sets
   b. train the model using training. produce output or its statistics. 
   c. record the witheld output (make abstract, talk to Ryan). Standard way would be to record case 1 output for the model where case 1 is omitted. 
   d. do one more model run with nothing witheld. That will be the final model
   ModelBuilder: will try to have this be automatic if you provide the other pieces. Do this in such away that we have a full_data=True option that 
                 does not do cross-validation.

8. Save the model.

"""


def xvalid_fit(
    df_in,
    df_out,
    builder,
    nepochs=80,
    plot_folds=[],
    plot_locs=["cse", "bdl", "emm2", "jer", "rsl"],
    out_prefix="ann_diag",
):
    """Splits up the input by fold, witholding each fold in turn, building and training the model for each
    and then evaluating the witheld data
    """

    bad_plot_loc = [x for x in plot_locs if x not in builder.output_names]
    if len(bad_plot_loc) > 0:
        raise ValueError(f"Some plot locations are not in the output: {bad_plot_loc}")

    df_out.to_csv(
        f"{out_prefix}_xvalid_ref_out.csv",
        float_format="%.3f",
        date_format="%Y-%m-%dT%H:%M",
        header=True,
        index=True,
    )

    # This constructs inputs with antecedent/lagged values, keeping the cases separate
    # This only makes sense for the variables that are ANN predictors, as determined by input_names
    # not variables "datetime," "fold" and "case". Those are preserved.
    inputs_lagged = builder.calc_antecedent_preserve_cases(df_in)

    # Lagging causes some data at the beginning to be trimmed. This performs the same trim on the outputs
    outputs_trim = df_out.loc[inputs_lagged.index, :]

    # Pre-construct data structure that will receive predictions for data left out.
    outputs_xvalid = pd.DataFrame().reindex_like(outputs_trim)
    outputs_xvalid[["datetime", "case", "fold"]] = outputs_trim[
        ["datetime", "case", "fold"]
    ]

    # Dictionary that will store predictions of all times from all the folds (key will be the fold identifier)
    # This includes biased predictions made of data in fold ifold=N during folds when that data are used for fitting
    all_outs = {}
    xvalid_outs = []

    for ifold in df_in.fold.unique():
        print(f"Starting fit for fold {ifold}")
        fit_in = inputs_lagged.loc[inputs_lagged.fold != ifold, :]

        # fit_in = fit_in.loc[fit_in.ndo_lag0 < 70000.,:]
        # fit_in = fit_in.loc[fit_in.sac_flow_lag0 < 70000.,:]
        fit_out = df_out.loc[fit_in.index, builder.output_names]
        test_in = inputs_lagged.loc[inputs_lagged.fold == ifold, :]
        test_out = df_out.loc[test_in.index, builder.output_names]

        print(
            f"ifold={ifold} # train input data rows = {fit_in.shape[0]} # train out rows = {fit_out.shape[0]}"
        )
        print(
            f"ifold={ifold} # test input data rows = {test_in.shape[0]} # test out rows = {test_out.shape[0]}"
        )

        input_layers = builder.input_layers()

        # The preprocessing layers are scaled using the full dataset so that the scaling is the same every time

        testdt = test_in.datetime
        ann = builder.build_model(input_layers, df_in)

        idx = pd.IndexSlice
        # These libraries separate the inputs into individual dataframes matching the input names
        fit_in = builder.df_by_feature_and_time(fit_in).drop(
            ["datetime", "case", "fold"], level="var", axis=1
        )
        fit_in = {
            name: fit_in.loc[:, idx[name, :]].droplevel("var", axis=1)
            for name in builder.input_names
        }

        testdt = test_in.datetime
        test_in = builder.df_by_feature_and_time(test_in).drop(
            ["datetime", "case", "fold"], level="var", axis=1
        )
        test_in = {
            name: test_in.loc[:, idx[name, :]].droplevel("var", axis=1)
            for name in builder.input_names
        }

        history, ann = builder.fit_model(
            ann, fit_in, fit_out, test_in, test_out, nepochs=nepochs
        )
        test_pred = ann.predict(test_in)
        outputs_xvalid.loc[test_out.index, builder.output_names] = (
            test_pred  # should be numpy array
        )
        print("outputs_xvalid")
        print(outputs_xvalid)
        checkname = f"{out_prefix}_check_{ifold}.csv"
        print(f"Completed fit for fold {ifold}. Writing check point csv to {checkname}")

        outputs_xvalid.to_csv(
            checkname,
            float_format="%.3f",
            date_format="%Y-%m-%dT%H:%M",
            header=True,
            index=True,
        )

        if (plot_folds == "all") or (ifold in plot_folds):
            height = 2.5 + 1.5 * len(plot_locs)
            fig, axes = plt.subplots(
                len(plot_locs),
                sharex=True,
                figsize=(8, height),
                layout="constrained",
                squeeze=False,
            )
            for iplot, loc in enumerate(plot_locs):
                ax = axes[iplot, 0]
                iloc = builder.output_names.index(loc)
                ax.plot(testdt, test_out.loc[:, loc])
                ax.plot(testdt, test_pred[:, iloc])
                if iplot == 0:
                    ax.legend(["model", "surrogate"])
                ax.set_title(f"Xvalid Fold: {ifold} Location: {loc}")
                ax.tick_params(axis="x", labelrotation=60)
            fig.savefig(f"{out_prefix}_{ifold}.png")

    outputs_xvalid.to_csv(
        f"{out_prefix}_xvalid.csv",
        float_format="%.3f",
        date_format="%Y-%m-%dT%H:%M",
        header=True,
        index=True,
    )


if __name__ == "__main__":

    # test_xvalid_mlp()
    test_xvalid_gru()
