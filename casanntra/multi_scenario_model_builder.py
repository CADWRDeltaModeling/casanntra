from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, regularizers, layers
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Reshape, Concatenate
from keras.models import load_model

from casanntra.model_builder import (ModelBuilder, UnscaleLayer, ScaledMaskedMAE, ScaledMaskedMSE, masked_mae, masked_mse, ModifiedExponentialDecayLayer)

class MultiScenarioModelBuilder(ModelBuilder):
    def __init__(self, input_names, output_names, ndays=90, **kwargs):
        super().__init__(input_names, output_names, ndays=ndays)
        self.ntime = ndays
        self.ndays = ndays
        self.nwindows = 0
        self.window_length = 0
        self.reverse_time_inputs = False  
        self.trunk_spec: List[Dict] = []
        self.branch_spec: Dict = {}
        self.include_source_branch: bool = True
        self.head_activation: str = "elu"
        self.init_targets_from_source: bool = True
        self.scenarios_cfg: List[Dict] = []      
        self.source_weight: float = 1.0
        self.target_weight_default: float = 1.0
        self.contrast_weight_default: float = 0.5
        self._supervised_keys: List[str] = []   
        self._contrast_keys: List[str] = []    
        self.transfer_type: str = "direct"  

    def set_builder_args(self, builder_args):
        """
        Parse per-step YAML builder_args.
        """
        super().set_builder_args(builder_args)

        tt = builder_args.get("transfer_type", None)
        tt = "direct" if tt in (None, "None") else str(tt).lower()
        if tt == "difference":
            tt = "direct"
        if tt not in ("direct", "contrastive"):
            raise ValueError(f"Unknown transfer_type: {tt}")
        
        self.transfer_type = tt

        self.trunk_spec = (builder_args.get("trunk_layers") or builder_args.get("base_layers"))
        self.branch_spec = builder_args.get("branch_layer", {"type": "GRU", "units": 16, "name": "feature3", "return_sequences": False, "trainable": True})

        self.include_source_branch = bool(builder_args.get("include_source_branch", True))
        self.head_activation = builder_args.get("head_activation", "elu")
        self.init_targets_from_source = bool(builder_args.get("init_targets_from_source", True))

        self.scenarios_cfg = builder_args.get("scenarios", []) or []
        self.source_weight = float(builder_args.get("source_weight", 1.0))
        self.target_weight_default = float(builder_args.get("target_weight", 1.0))
        self.contrast_weight_default = float(builder_args.get("contrast_weight", 0.5))

        self._supervised_keys = (["out_base_unscaled"] + [f"out_{sc['id']}_unscaled" for sc in self.scenarios_cfg])
        self._contrast_keys = [f"out_{sc['id']}_contrast_unscaled" for sc in self.scenarios_cfg]

    def requires_secondary_data(self) -> bool:
        return self.is_multi_scenario_step()

    def is_multi_scenario_step(self) -> bool:
        try:
            return (self.transfer_type == "contrastive" and isinstance(self.scenarios_cfg, list)
                and len(self.scenarios_cfg) > 0
            )
        except Exception:
            return False

    def num_outputs(self):
        return 1 + (len(self.scenarios_cfg) if self.is_multi_scenario_step() else 0)

    def map_prediction_keys_to_outputs(self, pred_keys):
        if not self.is_multi_scenario_step():
            return None
        return [k for k in self._supervised_keys if k in pred_keys]

    def _layer_cls(self, layer_type: str):
        lut = {"gru": GRU, "lstm": LSTM, "dense": Dense}
        return lut[layer_type.lower()]

    def _apply_trainable_flags(self, model: Model):
        for spec in self.trunk_spec:
            lname = spec.get("name")
            if lname and lname in {l.name for l in model.layers}:
                model.get_layer(lname).trainable = bool(spec.get("trainable", True))

        bname = self.branch_spec.get("name")
        if bname and bname in {l.name for l in model.layers}:
            model.get_layer(bname).trainable = bool(self.branch_spec.get("trainable", True))
        for sc in self.scenarios_cfg:
            sname = f"{self.branch_spec.get('name')}_{sc['id']}"
            if sname in {l.name for l in model.layers}:
                model.get_layer(sname).trainable = bool(self.branch_spec.get("trainable", True))

    def _build_trunk(self, x):
        z = x
        for i, spec in enumerate(self.trunk_spec):
            cls = self._layer_cls(spec["type"])
            name = spec.get("name", f"trunk_{i+1}")
            kw = {k: v for k, v in spec.items() if k not in {"type", "name", "trainable"}}
            if cls in (GRU, LSTM):
                kw.setdefault("activation", "sigmoid")
                kw.setdefault("return_sequences", i < len(self.trunk_spec) - 1)
            layer = cls(name=name, **kw)
            z = layer(z)
        return z

    def _build_branch_and_head(self, feat, head_id: str, base_branch: bool) -> Dict[str, tf.Tensor]:
        cls = self._layer_cls(self.branch_spec["type"])
        branch_name = self.branch_spec.get("name", "branch")
        if not base_branch:
            branch_name = f"{branch_name}_{head_id}" 

        branch_kw = {k: v for k, v in self.branch_spec.items() if k not in {"type", "name", "trainable"}}
        if cls in (GRU, LSTM):
            branch_kw.setdefault("activation", "sigmoid")
            branch_kw.setdefault("return_sequences", False)
        branch_layer = cls(name=branch_name, **branch_kw)
        if len(feat.shape) == 3:   
            z = branch_layer(feat)
        else:
            if cls in (GRU, LSTM):
                feat3 = layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(feat) 
                z = branch_layer(feat3)  
            else:
                z = branch_layer(feat)

        outdim = len(self.output_names)
        head_dense_name = f"head_{head_id}_scaled"
        y_scaled = Dense(outdim, activation=self.head_activation, name=head_dense_name)(z)
        y_unscaled = UnscaleLayer(list(self.output_names.values()), name=f"out_{head_id}_unscaled")(y_scaled)

        return {"branch_name": branch_name, "head_dense_name": head_dense_name, "out_name": f"out_{head_id}_unscaled", "y_unscaled": y_unscaled, "y_scaled": y_scaled}

    def _load_previous_model(self) -> Optional[Model]:
        if self.load_model_fname is None:
            return None
        
        print(f"[MultiScenario] Loading base model from: {self.load_model_fname}")
        base_model = load_model(self.load_model_fname + ".h5", custom_objects=self.custom_objects)

        try:
            base_model.load_weights(self.load_model_fname + ".weights.h5")
        except Exception:
            pass 
        return base_model

    def _try_copy_layer_weights(self, src_model: Model, dst_model: Model, src_name: str, dst_name: str):
        if src_name in {l.name for l in src_model.layers} and dst_name in {l.name for l in dst_model.layers}:
            try:
                dst_model.get_layer(dst_name).set_weights(src_model.get_layer(src_name).get_weights())
                print(f"[weights] copied {src_name} -> {dst_name}")
            except Exception as e:
                print(f"[weights] skip copy {src_name}->{dst_name}: {e}")

    def _init_all_weights(self, prev: Optional[Model], ann: Model):
        if prev is None:
            return

        for spec in self.trunk_spec:
            lname = spec.get("name")
            if lname:
                self._try_copy_layer_weights(prev, ann, lname, lname)

        base_branch_name = self.branch_spec.get("name")
        if base_branch_name:
            self._try_copy_layer_weights(prev, ann, base_branch_name, base_branch_name)

        possible_prev_heads = ["head_base_scaled", "out_scaled", "out_target_scaled"]
        dst_head = "head_base_scaled"
        for src_head in possible_prev_heads:
            if src_head in {l.name for l in prev.layers}:
                self._try_copy_layer_weights(prev, ann, src_head, dst_head)
                break

        if self.init_targets_from_source and base_branch_name:
            for sc in self.scenarios_cfg:
                sid = sc["id"]
                scen_branch = f"{base_branch_name}_{sid}"
                self._try_copy_layer_weights(ann, ann, base_branch_name, scen_branch)
                self._try_copy_layer_weights(ann, ann, "head_base_scaled", f"head_{sid}_scaled")

    def build_model(self, input_layers, input_data):
        prepro = self.prepro_layers(input_layers, input_data)
        expanded = [Reshape((self.ndays, 1))(t) for t in prepro]
        x = Concatenate(axis=-1, name="stacked")(expanded)

        feat = self._build_trunk(x)
        outputs = {}
        base_pack = self._build_branch_and_head(feat, head_id="base", base_branch=True)
        outputs[base_pack["out_name"]] = base_pack["y_unscaled"]

        if self.is_multi_scenario_step():
            for sc in self.scenarios_cfg:
                sid = sc["id"]
                scen_pack = self._build_branch_and_head(feat, head_id=sid, base_branch=False)
                outputs[scen_pack["out_name"]] = scen_pack["y_unscaled"]
                outputs[f"out_{sid}_contrast_unscaled"] = layers.Subtract(
                    name=f"out_{sid}_contrast_unscaled"
                )([scen_pack["y_unscaled"], base_pack["y_unscaled"]])

        ann = Model(inputs=input_layers, outputs=outputs, name="multi_scenario_model")

        prev = self._load_previous_model()
        if prev is not None:
            self._init_all_weights(prev, ann)

        self._apply_trainable_flags(ann)

        print(ann.summary())
        return ann

    def pool_and_align_cases(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        if not dataframes or len(dataframes) < 1:
            raise ValueError("pool_and_align_cases expects at least one DataFrame (Base).")

        all_idx = (pd.concat([df[["case", "datetime"]] for df in dataframes]).drop_duplicates().sort_values(["case", "datetime"]))
        aligned = [all_idx.merge(df, on=["case", "datetime"], how="left") for df in dataframes]
        input_cols = list(self.input_names)
        output_cols = list(self.output_names)

        merged_inputs = aligned[0][["case", "datetime"] + input_cols].copy()
        for df in aligned[1:]:
            for col in input_cols:
                if col in df.columns:
                    merged_inputs[col] = merged_inputs[col].combine_first(df[col])

        final = []
        for i, df in enumerate(aligned):
            out_df = merged_inputs.copy()
            for col in output_cols:
                out_df[col] = df[col] if col in df.columns else np.nan
            for extra in ["model", "scene"]:
                if extra in df.columns and extra not in out_df.columns:
                    out_df[extra] = df[extra]
            final.append(out_df)

        return final

    def fit_model(
        self,
        ann: Model,
        fit_input: Dict[str, np.ndarray],
        fit_output,               
        test_in: Dict[str, np.ndarray],
        test_out,                   
        init_train_rate: float,
        init_epochs: int,
        main_train_rate: float,
        main_epochs: int):

        output_scales = list(self.output_names.values())

        if not self.is_multi_scenario_step():
            ann.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
                        loss={"out_base_unscaled": ScaledMaskedMAE(output_scales)},
                        metrics={"out_base_unscaled": [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)]},run_eagerly=False)

            history = ann.fit(
                fit_input,
                {"out_base_unscaled": fit_output[self.output_list()].values if isinstance(fit_output, pd.DataFrame) else fit_output[0][self.output_list()].values},
                epochs=init_epochs,
                batch_size=64,
                validation_data=(test_in, {"out_base_unscaled": test_out[self.output_list()].values if isinstance(test_out, pd.DataFrame) else test_out[0][self.output_list()].values}),
                verbose=2,
                shuffle=True)

            if main_epochs and main_epochs > 0:
                ann.compile(
                    optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate, clipnorm=0.5),
                    loss={"out_base_unscaled": ScaledMaskedMAE(output_scales)},
                    metrics={"out_base_unscaled": [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)]},
                    run_eagerly=False)
                
                history = ann.fit(
                    fit_input,
                    {"out_base_unscaled": fit_output[self.output_list()].values if isinstance(fit_output, pd.DataFrame) else fit_output[0][self.output_list()].values},
                    epochs=main_epochs,
                    batch_size=64,
                    validation_data=(test_in, {"out_base_unscaled": test_out[self.output_list()].values if isinstance(test_out, pd.DataFrame) else test_out[0][self.output_list()].values}),
                    verbose=2,
                    shuffle=True)
            
            return history, ann

        if not isinstance(fit_output, list) or len(fit_output) < 1:
            raise ValueError("Multi-scenario step expects fit_output as [Base, Scen1, Scen2, ...].")

        base_train = fit_output[0][self.output_list()].values
        base_test = test_out[0][self.output_list()].values

        train_y = {"out_base_unscaled": base_train}
        test_y = {"out_base_unscaled": base_test}

        loss_dict = {"out_base_unscaled": ScaledMaskedMAE(output_scales)}
        metrics_dict = {"out_base_unscaled": [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)]}
        loss_wts = {"out_base_unscaled": float(self.source_weight)}

        for i, sc in enumerate(self.scenarios_cfg, start=1):
            sid = sc["id"]
            scen_train = fit_output[i][self.output_list()].values
            scen_test = test_out[i][self.output_list()].values
            train_y[f"out_{sid}_unscaled"] = scen_train
            test_y[f"out_{sid}_unscaled"]  = scen_test
            loss_dict[f"out_{sid}_unscaled"] = ScaledMaskedMAE(output_scales)
            metrics_dict[f"out_{sid}_unscaled"] = [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)]
            tgt_w = float(sc.get("target_weight", self.target_weight_default))
            loss_wts[f"out_{sid}_unscaled"] = tgt_w
            contrast_train = scen_train - base_train
            contrast_test = scen_test - base_test
            nan_mask_train = np.isnan(scen_train) | np.isnan(base_train)
            nan_mask_test = np.isnan(scen_test) | np.isnan(base_test)
            contrast_train[nan_mask_train] = np.nan
            contrast_test[nan_mask_test] = np.nan
            train_y[f"out_{sid}_contrast_unscaled"] = contrast_train
            test_y[f"out_{sid}_contrast_unscaled"]  = contrast_test
            loss_dict[f"out_{sid}_contrast_unscaled"] = ScaledMaskedMAE(output_scales)
            metrics_dict[f"out_{sid}_contrast_unscaled"] = [masked_mae, masked_mse]
            ctr_w = float(sc.get("contrast_weight", self.contrast_weight_default))
            loss_wts[f"out_{sid}_contrast_unscaled"] = ctr_w

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
            loss=loss_dict,
            loss_weights=loss_wts,
            metrics=metrics_dict,
            run_eagerly=False)

        history = ann.fit(
            fit_input,
            train_y,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_in, test_y),
            verbose=2,
            shuffle=True)

        if main_epochs and main_epochs > 0:
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate, clipnorm=0.5),
                loss=loss_dict,
                loss_weights=loss_wts,
                metrics=metrics_dict,
                run_eagerly=False)
            
            history = ann.fit(
                fit_input,
                train_y,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_in, test_y),
                verbose=2,
                shuffle=True)
            
        return history, ann