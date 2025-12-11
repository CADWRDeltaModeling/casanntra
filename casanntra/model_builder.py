import pandas as pd
from casanntra.read_data import read_data
from casanntra.scaling import ModifiedExponentialDecayLayer
from keras.models import load_model
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.keras.layers import Dense, Input, Layer, Multiply, Lambda
from tensorflow.keras.layers.experimental.preprocessing import Normalization, IntegerLookup, Rescaling #CategoryEncoding
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.layers import Layer
 
class ScaledMaskedMAE(tf.keras.losses.Loss):
    """Pickle-safe, TensorFlow-serializable custom loss function for scaled MAE."""

    def __init__(self, output_scales, name="scaled_mae"):
        super().__init__(name=name)
        self.scales_tensor = tf.constant(output_scales, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true_scaled = y_true / self.scales_tensor
        y_pred_scaled = y_pred / self.scales_tensor

        # ✅ Replace NaNs in `y_true` with corresponding predictions
        y_true_scaled = tf.where(tf.math.is_nan(y_true_scaled), y_pred_scaled, y_true_scaled)

        absolute_differences = tf.abs(y_true_scaled - y_pred_scaled)

        return tf.reduce_mean(absolute_differences)
    
    def get_config(self):
        """Ensures Keras can serialize this loss with its name."""
        config = super().get_config()
        config.update({"output_scales": self.scales_tensor.numpy().tolist()})
        return config

class ScaledMaskedMSE(tf.keras.losses.Loss):
    """Pickle-safe, TensorFlow-serializable custom loss function for scaled MSE."""

    def __init__(self, output_scales, name="scaled_mse"):
        super().__init__(name=name)
        self.scales_tensor = tf.constant(output_scales, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true_scaled = y_true / self.scales_tensor
        y_pred_scaled = y_pred / self.scales_tensor

        # ✅ Replace NaNs in `y_true` with corresponding predictions
        y_true_scaled = tf.where(tf.math.is_nan(y_true_scaled), y_pred_scaled, y_true_scaled)

        squared_diff = tf.square(y_true_scaled - y_pred_scaled)

        return tf.reduce_mean(squared_diff)

    def get_config(self):
        """Ensures Keras can serialize this loss with its name."""
        config = super().get_config()
        config.update({"output_scales": self.scales_tensor.numpy().tolist()})
        return config


def scaled_masked_mse(output_scales):
    scales = tf.constant(output_scales, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_scaled = y_true / scales
        y_pred_scaled = y_pred / scales

        mask = ~tf.math.is_nan(y_true_scaled)
        diff = tf.boolean_mask(y_true_scaled - y_pred_scaled, mask)
        squared_diff = tf.square(diff)

        return tf.reduce_mean(squared_diff)


##@tf.keras.utils.register_keras_serializable
def masked_mae(y_true, y_pred):
    # Mask NaN values, replace by 0
    y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)

    # Calculate absolute differences
    absolute_differences = tf.abs(y_true - y_pred)

    # Compute the mean, ignoring potential NaN values (if any remain after replacement)
    mae = tf.reduce_mean(absolute_differences)

    return mae


# @tf.keras.utils.register_keras_serializable
def masked_mse(y_true, y_pred):
    # Mask NaN values, replace by 0
    y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)

    # Calculate absolute differences
    absolute_differences = tf.square(y_true - y_pred)

    # Compute the mean, ignoring potential NaN values (if any remain after replacement)
    mae = tf.reduce_mean(absolute_differences)

    return mae




class UnscaleLayer(Layer):
    def __init__(self, output_scales, **kwargs):
        super().__init__(**kwargs)
        self.output_scales_init = output_scales

    def build(self, input_shape):
        self.output_scales = self.add_weight(
            name="output_scales",
            shape=(1, len(self.output_scales_init)),
            initializer=tf.constant_initializer(self.output_scales_init),
            trainable=False,
        )

    def call(self, inputs):
        return inputs * self.output_scales

    def get_config(self):
        config = super().get_config()
        config.update({"output_scales": self.output_scales_init})
        return config



def mean_squared_error2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))


class ModelBuilder(object):
    def __init__(self, input_names, output_names, ndays=90):
        """Base class for model builders, now handling custom object registration."""
        self.input_names = input_names
        self.output_names = output_names  # Dict of {output_name: scale_factor}
        # time/lag configuration defaults used by downstream utilities
        self.ndays = ndays
        self.ntime = ndays               # synonym used in some builders
        self.nwindows = 0                # no aggregated windows by default
        self.window_length = 0           # window duration (unused when nwindows==0)
        self.reverse_time_inputs = False # forward chronological order for RNNs
        self.load_model_fname = None  # Used for transfer learning
        self.builder_args = {}        # Used for per-step configuration See process_config
        self.ann_output_name = 'output_scaled'  # default for single output
        self.ann_output_name_map = None  # default for multiple outputs
        output_scales = list(self.output_names.values())
        #  Centralized custom object registration
        self.custom_objects = {"UnscaleLayer": UnscaleLayer,
                               "StackLayer": StackLayer,
                               "ModifiedExponentialDecayLayer": ModifiedExponentialDecayLayer,
                               "masked_mae": masked_mae,
                               "masked_mse": masked_mse,
                               "scaled_masked_mae": ScaledMaskedMAE(output_scales),
                               "scaled_masked_mse": ScaledMaskedMSE(output_scales)}
        

    def set_builder_args(self, builder_args):
        """Allows builder_args to be updated dynamically between steps. No default implementation"""
        self.builder_args = builder_args
        self.transfer_type = builder_args.get("transfer_type", None)

        # Handle output names dynamically
        if 'ann_output_name' in builder_args:
            self.ann_output_name = builder_args['ann_output_name']
        
        if 'ann_output_name_map' in builder_args:
            self.ann_output_name_map = builder_args['ann_output_name_map']        

    def num_outputs(self):
        """Returns the number of ANN output tensors produced if it is a list of tensors. 
           This matters in transfer learning if the outputs include both an estimate of the 
           main data and an estimate of the difference during transfer. 
           Typical numbers will be 1 or 2, not "15".
        """
        return 1  # Default case, subclasses may override

    def requires_secondary_data(self):
        """Determines whether a second dataset is required for training."""
        return False  # Default case (single-output models)

    def register_custom_object(self, name, obj):
        """Allows subclasses to register additional custom objects."""
        self.custom_objects[name] = obj

    def load_existing_model(self):
        """Handles model loading and ensures all required custom objects are registered."""
        if self.load_model_fname is None:
            return None  # No model to load
        
        print(f"Loading model from {self.load_model_fname} with registered custom objects.")

        #print(f"Custom objects: {self.custom_objects}")
        base_model = load_model(self.load_model_fname+".h5", custom_objects=self.custom_objects)
        base_model.load_weights(self.load_model_fname+".weights.h5")
        return base_model


    def _create_unscaled_layer(self, scaled_output):
        """Creates an unscaled version of an existing output layer for inference purposes."""
        output_scales = list(self.output_names.values())  # Extract scale factors

        #  Unscaled output for inference
        return UnscaleLayer(output_scales, name="unscaled_output")(scaled_output)



    def raw_data_to_features(self, data):
        """Converts the raw data from the model to the features named by the model
        as discoverable using  feature_names()
        """
        features = self.feature_names()
        return data[features]

    def feature_names(self):
        return self.input_names

    def output_list(self):
        return list(self.output_names)

    def feature_dim(self, feature):
        """Returns the dimension of feature, typically lags (individual or aggregated)"""
        return self.ntime

    def df_by_feature_and_time(self, df):
        """Convert a dataset with a single index with var_lag as column names and convert to MultiIndex with (var,ndx)
        This facilitates queries that select only lags or only variables. As a side effect this routine will store
        the name of the active lags for each feature, corresponding to the number of lags in the dictionary num_feature_dims)
        into the module variable lag_features.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted

        Returns
        -------
        df_var : A copy of DataFrame with multiIndex based on feature,lag  (e.g. 'sac','4d')
        """

        df = df.copy()
        indextups = []


        for col in list(df.columns):
            var = col
            lag = ""
            for key in self.feature_names():
                if key in col:
                    var = key
                    lag = col.replace(key, "").replace("_", "")
                    if lag is None or lag == "":
                        lag = "0d"
                    continue
            if var == "EC":
                lag = "0d"

            indextups.append((var, lag))

        ndx = pd.MultiIndex.from_tuples(indextups, names=("var", "lag"))
        df.columns = ndx
        names = self.feature_names()
        return df


    def prepro_layers(self, inp_layers, df):
        """ Create the preprocessing layers, one per location, which will be concatenated later.
            This function performs any sort of special scaling. Here the superclass is overridden.

            Scaling for Bay-Delta problems is inherantly custom. Prior work has proceeded on the 
            hope that one can use a "best practice" like Normalize (mean and std). These are 
            questionably appropriate because the flow variables saturate an order of magnitude lower than
            their upper limits. (e.g. above 40,000cfs for Sac). 

            For DSM2, which can't model low X2, the challenge is to try to saturate at high outflow and 
            scale the inflows so that the salinity-affecting range is some reasonable (0-1 ish) range.
             
            Scaling rivers with volatile statistics does not achieve this. For instance, Sac River 
            flow maximum depends wildly on the choice of year, and if you are doing transfer learning
            from one group of years to another this is erratic.  The range could be as much as 300,000 
            which compresses the portion pertinent for salinity management into Q' < 0.1. Above this 
            number, DSM2 doesn't have any variables that are sensitive to the inputs ... so in a sense
            you might even be better off omitting these data since they don't help the solve.

            One of the original solutions to this was to just use a threshold like 40,000cfs for the Sacramento River 
            and 10,000cfs for the SJR. It works well in DSM2. The required saturation above 70kcfs seems to happen
            by good fortune. 
             
            When you anticipate transferring from DSM2 to SCHISM/RMA you might plan for the big difference in
            the ability of those to models to reproduce X2 < 55km, which means that they have meaningful 
            gradients for training under much larger flows than DSM2 does. In fact they might (?) be hobbled 
            by a 40kcfs saturation. An example custom function based on modified exponential decay is 
            available in scaling.py and an example of its use may be found, sometiems commented, in fit_from_config.py. 
    
        """
        layers = []
        names = self.feature_names()
        if len(names) != len(inp_layers ): 
            raise ValueError("Inconsistency in number of layers between inp_layers and feature names")
        thresh = 40000.
        dims = {x: self.feature_dim(x) for x in names} 

        for feature in self.feature_names():
            station_df = df.loc[:,feature]
            xinput = inp_layers[feature]
            prepro_name=f"{feature}_prepro" 
            if feature in ["dcc", "smscg"]:
                feature_layer = Rescaling(1.0,name=prepro_name)
            elif feature in [ "northern_flow", "sac_flow", "ndo"] and thresh is not None:
                # Define the model. Use the test_scaling file to refine parameters
                feature_layer = ModifiedExponentialDecayLayer(a=1.e-5, b=70000., name=prepro_name)
                #feature_layer = Rescaling(1 / thresh, name=prepro_name)  # Normalization(axis=None)
            elif feature == "sjr_flow" and thresh is not None:
                feature_layer = ModifiedExponentialDecayLayer(a=1.e-5, b=40000., name=prepro_name)
                # feature_layer = Rescaling(0.25 / thresh, name=prepro_name)  # Normalization(axis=None)         
            else:
                feature_layer = Normalization(axis=None,name=prepro_name)
                feature_layer.adapt(
                    station_df.to_numpy())
            layers.append(feature_layer(xinput))
        return layers

    def input_layers(self):
        """Takes a dataframe representing variables 
        and processes into a list of input layers. 
        The input should be the original data without lags normalization
        should be applied if you want normalization to be applied before
        the first ANN computational layer. The example implementation shows some ways to do it.
        This would likely be overridden for different choices, though it works for many
        """
        layers = {}
        names = self.feature_names()  

        for feature in self.feature_names():
            layers[feature] = Input(shape=(self.feature_dim(feature)), name=feature)

        
        return layers
    
    def register_custom_object(self, name, obj):
        """Allows subclasses to register additional custom objects."""
        self.custom_objects[name] = obj    

    def build_model(self, input_layers, input_data, add_unscaled_output=False):
        """ Builds out the architecture after input_layers through the compile. 
           The input_layers are created outside the model to ensure they follow the requested form
           Often the next stage will be to call prepro_layers as in the recursive example, 
           though this is not required
        """
        raise NotImplementedError("Must be implemented")

    
    def fit_model(self, model, fit_in, fit_out, test_in, test_out):
        """ Fits the built model 
        """
        raise NotImplementedError("Must be implemented")
    

    def calc_antecedent_preserve_cases(self, df, ndays=-1, nwindows=-1, window_length=-1,reverse=None):
        """
        Calculates lags for data with multiple cases so that dates alone may not be unique and lags can't be calculated naively by shifting rows
        Input df must have a "case" columns
        
        Parameters
        ----------
        df : DataFrame
            A dataframe that has a 'datetime' and 'cases' column and for which datetime,cases are a unique key
        
        ndays : int
            Number of individual day lags

        nwindows: int
            Number of aggregated windows of days, which will start preceding ndays
        
        window_length : int
            Duration of days of the windows

        reverse : bool
            Whether to return the lags in backward looking order starting with "now". The default for LSTM is no, 
        since this is part of the LSTM/GRU architecture. For the CalSim MLPs this is customarily True. The method
        currently raises an error if nwindows > 0 and reverse=False because this isn't plumbed out
        """

        if not "case" in df.columns:
            raise ValueError("No cases column")
        
        if reverse is None:
            reverse = self.reverse_time_inputs

        if not reverse and nwindows>0:
            raise NotImplementedError("Not implmented for non-reverse plus aggregation windows.")

        if ndays<0: 
            ndays = self.ndays
        if nwindows < 0:
            nwindows = self.nwindows
        if window_length < 0:
            window_length = self.window_length


        antecedes = []
        for case in df.case.unique():
            df_case = df.loc[df.case == case]
            
            antecedent = self.create_antecedent_inputs(df_case,ndays,window_length,nwindows,reverse)
            antecedes.append(antecedent)

        return pd.concat(antecedes,axis=0)

    def _time_fold(self, sub_df,dt):
        """ Calculates the grouping in time for a single case based on target length in time"""
        sub_df = sub_df.copy()
        start = sub_df.datetime.min()
        end = sub_df.datetime.max()
        nbin = pd.to_timedelta(end-start)//pd.to_timedelta(dt)
        sub_df['index_bounds'] = pd.cut(sub_df.index,bins=nbin,labels=False)
        return sub_df.index_bounds

    def xvalid_time_folds(self, data, target_fold_len='180d',split_in_out=True, reverse=False):
        """ Calculates cross-validation folds for datframe data that assume there are cases and that each 
            case is divided approximately into chunks of length target_fold_len

            Parameters
            ----------

            data : pd.DataFrame
                An incoming dataframe that has cases and dates. Within a case, dates are unique and monotone

            target_fold_len: pandas freq like '180d'
                A length, typically smaller than the cases, that will be used to create smaller xvalidation folds. For
            instance, the cases might be up to 2 years long and target_fold_len='180d'.

            split_in_out: bold
                Whether to return input and output separately. Helps reduce bookkeeping

            reverse: bool 
                Whether to return lags in reverse chronicalogical order, something the CalSim models do 

        """
        data = data.copy()
        if isinstance(target_fold_len,str):
            dt = pd.to_timedelta(target_fold_len)

        #data = data.reset_index(drop=False)  # datetime and case are columns
        data['time_fold'] = -1
        if "case" not in data:
            data["case"] = 1


        for case in data.case.unique():
    
            case_df = data.loc[data.case == case]
            case_df.loc[:,'time_fold'] =self._time_fold(case_df,dt)
            data.update(case_df)

        data['fold'] = data.groupby(['case','time_fold']).ngroup()
        
        if split_in_out:
            df_in = data[["datetime","case","fold"]+self.input_names]      # isolate input
            df_out = data[["datetime","case","fold"]+list(self.output_names)]    # isolate output
            return df_in,df_out
        else:
            return data


    def create_antecedent_inputs(self, df, ndays=-1, window_length=-1, nwindows=-1, reverse=None):
        """
        Expands a dataframe to include lagged data.
        Each column of the input dataframe is expanded to:
        * ndays individual lagged days,  (current day + (ndays-1) previous days)
        * nwindows values of averaged data over non-overlapping blocks of window_length days per block
        The lags are ordered from most recent to farthest past, and the averaged blocks are from the days that precede the individual.

        Classic CalSIM is ndays=8, window_length=14, nwindows=5

        Returns
        -------

            df_x : pd.DataFrame
            A dataframe with input columns labeled like orig_lag1 and orig_ave1 etc for the
            individual and aggregated

        """
        
        if ndays < 0: 
            ndays = self.ndays
        if nwindows < 0:
            nwindows = self.nwindows
        if window_length < 0:
            window_length = self.window_length
        if reverse is None: reverse = self.reverse_time_inputs

        preserve_cols = [x for x in ["datetime", "case", "fold"] if x in df.columns]
        df2 = df[preserve_cols].copy()
        df = df.drop(preserve_cols, axis=1)
        orig_index = df.index

        if not reverse and nwindows > 0:
            raise NotImplementedError("Not Implemented.")

        if reverse:
            arr1 = [df.shift(n) for n in range(ndays)]
        else:
            arr1 = [df.shift(n) for n in reversed(range(ndays))]

        if nwindows > 0:
            dfr = df.rolling(window=window_length, min_periods=window_length).mean()
            arr2 = [dfr.shift(window_length * n + ndays) for n in range(nwindows)]                    
        else:
            arr2 = []

        df_x = pd.concat(arr1 + arr2, axis=1).dropna()  # nsamples, nfeatures


        # Adjust column names
        new_columns = [] #preserve_cols

        for n in range(ndays):
            for col in df.columns:
                if col not in preserve_cols:
                    new_columns.append(col + '_lag{}'.format(n))
        for n in range(nwindows):
            for col in df.columns:
                if col not in preserve_cols: 
                    new_columns.append(col + '_avg{}'.format(n))

        df_x.columns = new_columns
        
        df_x = df2.join(df_x, how="right")
        return df_x



    def wrap_with_unscale_layer2(self, trained_model):
        output_scales = list(self.output_names.values())

        if isinstance(trained_model.output, dict):
            output_dict = trained_model.output
        elif isinstance(trained_model.output, list):
            output_dict = {tensor.name.split('/')[0]: tensor for tensor in trained_model.output}
        else:
            output_dict = {"output_scaled": trained_model.output}

        unscaled_outputs = {}
        for key, tensor in output_dict.items():
            safe_key = key.replace("/", "_") + "_unscaled"
            unscaled_outputs[safe_key] = UnscaleLayer(output_scales, name=safe_key)(tensor)

        wrapped_model = Model(inputs=trained_model.input, outputs=unscaled_outputs)
        
        return wrapped_model
    
    def is_multi_scenario_step(self) -> bool:
        try:
            sc = self.builder_args.get("scenarios", None)
            return isinstance(sc, list) and len(sc) > 0
        except Exception:
            return False

    def map_prediction_keys_to_outputs(self, pred_keys):
        return None  # Multi-scenario builder will override



class StackLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(StackLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.stack(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


#########################################################

class GRUBuilder2(ModelBuilder):
    """
    This model builder is the current (2025-01-20) standard recursive model
    """

    def __init__(self,input_names,output_names,ndays,load_model_fname=None):
        super().__init__(input_names,output_names,load_model_fname)
        self.ntime = ndays                  
        self.ndays = ndays                  # Number of individual days
        self.nwindows = 0                   # Number of windows, always zero for recursive
        self.window_length = 0              # Length of each windows
        self.reverse_time_inputs = False    # Reorder days looking backward. 
                                            # False for recursive, True (by convention) for 
        self.register_custom_object("ModifiedExponentialDecayLayer", ModifiedExponentialDecayLayer)
    
        


    def build_model(self,input_layers, input_data):
        """ Build or load the model architecture. Or load an old one and extend."""
        
        do_load = self.load_model_fname is not None
        print(f"do_load={do_load}, load_model_fname={self.load_model_fname}")
        if do_load:
            print(f"Loading from {self.load_model_fname} for refinement")
            ann = load_model(self.load_model_fname,custom_objects=self.custom_objects)
            ann.load_weights(self.load_model_fname.replace(".h5",".weights.h5"))

            print(ann.summary())
            return ann
        else:
            print(f"Creating from scratch")
            prepro_layers = self.prepro_layers(input_layers,input_data)          
            #x = StackLayer(name="stack_layer")(prepro_layers)
            x = Lambda(lambda inputs: tf.stack(inputs, axis=-1), name="StackLambda")(input_layers)
            x = tf.keras.layers.concatenate(prepro_layers)
            x = layers.LSTM(units=32, return_sequences=True, 
                                activation='sigmoid',name='gru_1')(x)
            x = layers.LSTM(units=16, return_sequences=False, 
                                activation='sigmoid',name='gru_2')(x)
            x = layers.Flatten()(x)
            
            outdim = len(self.output_names)
            # The regularization is unknown
            scaled_output = layers.Dense(units=outdim, name = "out_scaled", activation='elu',
                                kernel_regularizer = regularizers.l1_l2(l1=0.0001,l2=0.0001))(x)

            # explicitly add unscaling directly in base model
            unscaled_output = UnscaleLayer(list(self.output_names.values()), name="out_unscaled")(scaled_output)

            ann = Model(inputs=input_layers, outputs=unscaled_output)

            print(ann.summary())

        return ann

    def get_loss_function(self):
        """Returns the correct loss function based on transfer_type."""
        return "mae"  # Default case (single-output models)


    def fit_model(self, ann, fit_input, fit_output, test_input, test_output, init_train_rate, init_epochs, main_train_rate, main_epochs):
        """Custom fit_model that supports staged learning and multi-output cases."""

        #  Get loss function dynamically
        loss_function = self.get_loss_function()

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=1.0), 
            loss=loss_function,
            metrics=['mae', 'mse'],
            run_eagerly=True
        )
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
        )

        #  Main training phase (slower learning rate)
        if main_epochs is not None and main_epochs > 0:
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate), 
                loss=loss_function,
                metrics=['mae', 'mse'],
                run_eagerly=False
            )
            history = ann.fit(
                fit_input,
                fit_output,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_output),
                verbose=2,
                shuffle=True
            )

        return history, ann


###########################################################################################

class MLPBuilder1(ModelBuilder):
    """ Example ModelBuilder using CalSim-like MLP. """

    def __init__(self,input_names,output_names,ndays,nwindows,window_length):
        super().__init__(input_names,output_names,ndays,nwindows,window_length)


    def build_model(self,input_layers, input_data):
        """ Builds the standard CalSIM ANN
            Parameters
            ----------
            input_layers : list  
            List of tf Input layers

            input_data: dataframe
            Full list of data, without lags at this point. This is used for calling prepro_layers for scaling.
        """        
        outdim = len(self.output_names)
        prepro_layers = self.prepro_layers(input_layers,input_data)   

        x = layers.Concatenate()(prepro_layers) 
        
        # First hidden layer 
        x = Dense(units=12, activation='linear', input_dim=x.shape[1], 
                  kernel_initializer="he_normal",
                  #kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden1")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Second hidden layer 
        x = Dense(units=8, activation='sigmoid', kernel_initializer="he_normal",
                  #kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden2")(x) 
        x = tf.keras.layers.BatchNormalization(name="batch_normalize")(x)

        # Output layer with 1 neuron per output
        output = Dense(units=outdim,name="ec",activation="relu")(x)
        ann = Model(inputs = input_layers, outputs = output)
        print(ann.summary())
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.002), 
            loss="mae", 
            metrics=['mean_absolute_error'],
            run_eagerly=True
        )

        return ann

    def fit_model(self,ann,fit_input,fit_output,test_input,test_output,nepochs=200):
        tf.config.run_functions_eagerly(True)
        #tf.data.experimental.enable_debug_mode()
        history = ann.fit(
                fit_input,fit_output, 
                validation_data=(test_input, test_output), 
                batch_size=32, 
                epochs=nepochs, 
                verbose=2,
                shuffle=True
                )
        
        return history, ann
