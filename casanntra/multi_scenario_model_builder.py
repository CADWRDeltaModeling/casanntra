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
        self.branch_layers: List[Dict] = []
        self.per_scenario_branch: bool = False
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
        self.head_plan: List[Dict] = []

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

        branch_layers_cfg = builder_args.get("branch_layers")
        if branch_layers_cfg is None:
            single_branch = builder_args.get("branch_layer")
            if single_branch:
                branch_layers_cfg = [single_branch]
        if branch_layers_cfg is None:
            branch_layers_cfg = []
        elif isinstance(branch_layers_cfg, dict):
            branch_layers_cfg = [branch_layers_cfg]
        self.branch_layers = branch_layers_cfg
        self.per_scenario_branch = bool(builder_args.get("per_scenario_branch", False))

        self.include_source_branch = bool(builder_args.get("include_source_branch", True))
        self.head_activation = builder_args.get("head_activation", "elu")
        self.init_targets_from_source = bool(builder_args.get("init_targets_from_source", True))

        self.scenarios_cfg = builder_args.get("scenarios", []) or []
        self.source_weight = float(builder_args.get("source_weight", 1.0))
        self.target_weight_default = float(builder_args.get("target_weight", 1.0))
        self.contrast_weight_default = float(builder_args.get("contrast_weight", 0.5))

        self.head_plan = self._build_head_plan()

        self._supervised_keys = [
            spec["out_name"] for spec in self.head_plan if spec["kind"] == "dense"
        ] or ["out_base_unscaled"]
        self._contrast_keys = [spec["out_name"] for spec in self.head_plan if spec["kind"] == "contrast"]

    def requires_secondary_data(self) -> bool:
        return self.transfer_type == "contrastive" and len(self.scenarios_cfg) > 0

    def is_multi_scenario_step(self) -> bool:
        try:
            return (self.transfer_type == "contrastive" and isinstance(self.scenarios_cfg, list)
                and len(self.scenarios_cfg) > 0
            )
        except Exception:
            return False

    def num_outputs(self):
        return len(self._supervised_keys) if self._supervised_keys else 1

    def map_prediction_keys_to_outputs(self, pred_keys):
        if not self.requires_secondary_data():
            return None
        return [k for k in self._supervised_keys if k in pred_keys]

    def _layer_cls(self, layer_type: str):
        lut = {"gru": GRU, "lstm": LSTM, "dense": Dense}
        return lut[layer_type.lower()]

    def _head_uses_branch(self, head_id: str, is_base: bool) -> bool:
        if not self.branch_layers:
            return False
        if is_base:
            return bool(self.include_source_branch)
        return bool(self.per_scenario_branch)

    def _build_head_plan(self) -> List[Dict]:
        plan: List[Dict] = []
        use_base_branch = self._head_uses_branch("base", True)

        plan.append(self._dense_spec(
            head_id="base",
            out_name="out_base_unscaled",
            data_key="base",
            loss_weight=float(self.source_weight),
            builder="branch" if use_base_branch else "shared",
            dense_name="head_base_scaled"))

        if self.transfer_type != "contrastive" or len(self.scenarios_cfg) == 0:
            return plan

        for sc in self.scenarios_cfg:
            sid = sc["id"]
            tgt_w = float(sc.get("target_weight", self.target_weight_default))
            use_branch = self._head_uses_branch(sid, False)
            plan.append(self._dense_spec(
                head_id=sid,
                out_name=f"out_{sid}_unscaled",
                data_key=sid,
                loss_weight=tgt_w,
                builder="branch" if use_branch else "shared",
                dense_name=f"head_{sid}_scaled"))
            ctr_w = float(sc.get("contrast_weight", self.contrast_weight_default))
            plan.append(self._contrast_spec(
                out_name=f"out_{sid}_contrast_unscaled",
                pos_head=sid,
                neg_head="base",
                loss_weight=ctr_w))

        return plan

    def _dense_spec(
        self,
        head_id: str,
        out_name: str,
        data_key: str,
        loss_weight: float,
        builder: str,
        dense_name: str,
    ) -> Dict:
        return {
            "kind": "dense",
            "head_id": head_id,
            "out_name": out_name,
            "data_key": data_key,
            "loss_weight": loss_weight,
            "builder": builder,
            "dense_name": dense_name,
        }

    def _contrast_spec(self, out_name: str, pos_head: str, neg_head: str, loss_weight: float) -> Dict:
        return {
            "kind": "contrast",
            "out_name": out_name,
            "pos_head": pos_head,
            "neg_head": neg_head,
            "loss_weight": loss_weight,
        }

    def _apply_trainable_flags(self, model: Model):
        existing = {l.name for l in model.layers}
        for spec in self.trunk_spec:
            lname = spec.get("name")
            if lname and lname in existing:
                model.get_layer(lname).trainable = bool(spec.get("trainable", True))
        for idx, spec in enumerate(self.branch_layers):
            base_name = spec.get("name", f"branch_{idx+1}")
            trainable_flag = bool(spec.get("trainable", True))
            if base_name in existing:
                model.get_layer(base_name).trainable = trainable_flag
            for sc in self.scenarios_cfg:
                sname = self._branch_layer_name(spec, sc["id"], idx)
                if sname in existing:
                    model.get_layer(sname).trainable = trainable_flag

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

    def _branch_layer_name(self, spec: Dict, head_id: str, idx: int) -> str:
        base_name = spec.get("name", f"branch_{idx+1}")
        if head_id == "base":
            return base_name
        return f"{base_name}_{head_id}"

    def _apply_branch_layer(self, tensor, layer, is_recurrent: bool):
        if is_recurrent:
            if len(tensor.shape) == 3:
                return layer(tensor)
            feat3 = layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(tensor)
            return layer(feat3)
        return layer(tensor)

    def _build_head(self, feat, head_id: str, use_branch: bool, dense_name: Optional[str] = None) -> Dict[str, tf.Tensor]:
        z = feat
        if use_branch and self.branch_layers:
            for idx, spec in enumerate(self.branch_layers):
                cls = self._layer_cls(spec["type"])
                layer_name = self._branch_layer_name(spec, head_id, idx)
                kw = {k: v for k, v in spec.items() if k not in {"type", "name"}}
                layer = cls(name=layer_name, **kw)
                z = self._apply_branch_layer(z, layer, cls in (GRU, LSTM))

        outdim = len(self.output_names)
        head_dense_name = dense_name or f"head_{head_id}_scaled"
        y_scaled = Dense(outdim, activation=self.head_activation, name=head_dense_name)(z)
        y_unscaled = UnscaleLayer(list(self.output_names.values()), name=f"out_{head_id}_unscaled")(y_scaled)

        return {"head_dense_name": head_dense_name, "out_name": f"out_{head_id}_unscaled", "y_unscaled": y_unscaled, "y_scaled": y_scaled}

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

    def _try_copy_layer_weights(self, src_model: Model, dst_model: Model, src_name: str, dst_name: str) -> bool:
        src_layers = {l.name for l in src_model.layers}
        dst_layers = {l.name for l in dst_model.layers}
        if src_name in src_layers and dst_name in dst_layers:
            try:
                dst_model.get_layer(dst_name).set_weights(src_model.get_layer(src_name).get_weights())
                print(f"[weights] copied {src_name} -> {dst_name}")
                return True
            except Exception as e:
                print(f"[weights] skip copy {src_name}->{dst_name}: {e}")
        return False

    def _init_all_weights(self, prev: Optional[Model], ann: Model):
        if prev is None:
            return

        for spec in self.trunk_spec:
            lname = spec.get("name")
            if lname:
                self._try_copy_layer_weights(prev, ann, lname, lname)

        for idx, spec in enumerate(self.branch_layers):
            base_name = spec.get("name", f"branch_{idx+1}")
            if base_name:
                self._try_copy_layer_weights(prev, ann, base_name, base_name)

        possible_prev_heads = ["head_base_scaled", "source_scaled", "target_scaled", "out_scaled", "out_target_scaled"]
        for src_head in possible_prev_heads:
            if self._try_copy_layer_weights(prev, ann, src_head, "head_base_scaled"):
                break

        if self.init_targets_from_source:
            for sc in self.scenarios_cfg:
                sid = sc["id"]
                self._try_copy_layer_weights(ann, ann, "head_base_scaled", f"head_{sid}_scaled")
                for idx, spec in enumerate(self.branch_layers):
                    base_name = spec.get("name", f"branch_{idx+1}")
                    if not base_name:
                        continue
                    self._try_copy_layer_weights(ann, ann, base_name, self._branch_layer_name(spec, sid, idx))

    def _copy_preprocessing_layers(self, prev: Optional[Model], ann: Model):
        if prev is None:
            return

        prev_layers = {l.name: l for l in prev.layers}
        ann_layer_names = {l.name for l in ann.layers}

        for feature in self.input_names:
            lname = f"{feature}_prepro"
            if lname not in prev_layers or lname not in ann_layer_names:
                continue
            try:
                ann_layer = ann.get_layer(lname)
                prev_layer = prev_layers[lname]
                ann_layer.set_weights(prev_layer.get_weights())
                ann_layer.trainable = prev_layer.trainable
                print(f"[prepro] copied {lname}")
            except Exception as exc:
                print(f"[prepro] skip copy {lname}: {exc}")

    def build_model(self, input_layers, input_data):
        prepro = self.prepro_layers(input_layers, input_data)
        expanded = [Reshape((self.ndays, 1))(t) for t in prepro]
        x = Concatenate(axis=-1, name="stacked")(expanded)

        feat = self._build_trunk(x)
        outputs = {}
        head_tensors: Dict[str, tf.Tensor] = {}
        for spec in self.head_plan:
            if spec["kind"] != "dense":
                continue
            use_branch = spec["builder"] == "branch"
            pack = self._build_head(feat, head_id=spec["head_id"], use_branch=use_branch, dense_name=spec.get("dense_name"))
            outputs[spec["out_name"]] = pack["y_unscaled"]
            head_tensors[spec["head_id"]] = pack["y_unscaled"]

        for spec in self.head_plan:
            if spec["kind"] != "contrast":
                continue
            outputs[spec["out_name"]] = layers.Subtract(name=spec["out_name"])([
                head_tensors[spec["pos_head"]],
                head_tensors[spec["neg_head"]],
            ])

        ann = Model(inputs=input_layers, outputs=outputs, name="multi_scenario_model")

        prev = self._load_previous_model()
        if prev is not None:
            self._init_all_weights(prev, ann)
            self._copy_preprocessing_layers(prev, ann)

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

        if not self.requires_secondary_data():
            return self._fit_direct(
                ann,
                fit_input,
                fit_output,
                test_in,
                test_out,
                init_train_rate,
                init_epochs,
                main_train_rate,
                main_epochs,
            )

        return self._fit_contrastive(
            ann,
            fit_input,
            fit_output,
            test_in,
            test_out,
            init_train_rate,
            init_epochs,
            main_train_rate,
            main_epochs,
        )

    def _fit_direct(
        self,
        ann: Model,
        fit_input: Dict[str, np.ndarray],
        fit_output,
        test_in: Dict[str, np.ndarray],
        test_out,
        init_train_rate: float,
        init_epochs: int,
        main_train_rate: float,
        main_epochs: int,
    ):
        output_scales = list(self.output_names.values())
        target_key = self._supervised_keys[0]
        train_block = (
            fit_output[self.output_list()].values
            if isinstance(fit_output, pd.DataFrame)
            else fit_output[0][self.output_list()].values
        )
        test_block = (
            test_out[self.output_list()].values
            if isinstance(test_out, pd.DataFrame)
            else test_out[0][self.output_list()].values
        )

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
            loss={target_key: ScaledMaskedMAE(output_scales)},
            metrics={target_key: [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)]},
            run_eagerly=False,
        )
        history = ann.fit(
            fit_input,
            {target_key: train_block},
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_in, {target_key: test_block}),
            verbose=2,
            shuffle=True,
        )

        if main_epochs and main_epochs > 0:
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate, clipnorm=0.5),
                loss={target_key: ScaledMaskedMAE(output_scales)},
                metrics={target_key: [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)]},
                run_eagerly=False,
            )
            history = ann.fit(
                fit_input,
                {target_key: train_block},
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_in, {target_key: test_block}),
                verbose=2,
                shuffle=True,
            )

        return history, ann

    def _fit_contrastive(
        self,
        ann: Model,
        fit_input: Dict[str, np.ndarray],
        fit_output,
        test_in: Dict[str, np.ndarray],
        test_out,
        init_train_rate: float,
        init_epochs: int,
        main_train_rate: float,
        main_epochs: int,
    ):
        train_arrays = self._extract_output_arrays(fit_output, label="fit_output")
        test_arrays = self._extract_output_arrays(test_out, label="test_out")

        output_scales = list(self.output_names.values())
        train_y: Dict[str, np.ndarray] = {}
        test_y: Dict[str, np.ndarray] = {}
        loss_dict: Dict[str, tf.keras.losses.Loss] = {}
        loss_wts: Dict[str, float] = {}
        metrics_dict: Dict[str, List] = {}
        dense_train: Dict[str, np.ndarray] = {}
        dense_test: Dict[str, np.ndarray] = {}

        for spec in self.head_plan:
            if spec["kind"] != "dense":
                continue
            data_key = spec["data_key"]
            train_arr = train_arrays[data_key]
            test_arr = test_arrays[data_key]
            train_y[spec["out_name"]] = train_arr
            test_y[spec["out_name"]] = test_arr
            dense_train[spec["head_id"]] = train_arr
            dense_test[spec["head_id"]] = test_arr
            loss_dict[spec["out_name"]] = ScaledMaskedMAE(output_scales)
            metrics_dict[spec["out_name"]] = [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)]
            loss_wts[spec["out_name"]] = float(spec["loss_weight"])

        for spec in self.head_plan:
            if spec["kind"] != "contrast":
                continue
            pos_train = dense_train[spec["pos_head"]]
            neg_train = dense_train[spec["neg_head"]]
            pos_test = dense_test[spec["pos_head"]]
            neg_test = dense_test[spec["neg_head"]]

            contrast_train = pos_train - neg_train
            contrast_test = pos_test - neg_test
            nan_mask_train = np.isnan(pos_train) | np.isnan(neg_train)
            nan_mask_test = np.isnan(pos_test) | np.isnan(neg_test)
            contrast_train[nan_mask_train] = np.nan
            contrast_test[nan_mask_test] = np.nan

            train_y[spec["out_name"]] = contrast_train
            test_y[spec["out_name"]] = contrast_test
            loss_dict[spec["out_name"]] = ScaledMaskedMAE(output_scales)
            metrics_dict[spec["out_name"]] = [masked_mae, masked_mse]
            loss_wts[spec["out_name"]] = float(spec["loss_weight"])

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
            loss=loss_dict,
            loss_weights=loss_wts,
            metrics=metrics_dict,
            run_eagerly=False,
        )
        history = ann.fit(
            fit_input,
            train_y,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_in, test_y),
            verbose=2,
            shuffle=True,
        )

        if main_epochs and main_epochs > 0:
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate, clipnorm=0.5),
                loss=loss_dict,
                loss_weights=loss_wts,
                metrics=metrics_dict,
                run_eagerly=False,
            )
            history = ann.fit(
                fit_input,
                train_y,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_in, test_y),
                verbose=2,
                shuffle=True,
            )

        return history, ann

    def _extract_output_arrays(self, outputs, label: str) -> Dict[str, np.ndarray]:
        if not isinstance(outputs, list) or len(outputs) < 1:
            raise ValueError(f"Contrastive step expects '{label}' as [Base, Scenario1, ...].")

        expected = 1 + len(self.scenarios_cfg)
        if len(outputs) < expected:
            raise ValueError(f"Expected at least {expected} elements in '{label}', got {len(outputs)}.")

        cols = self.output_list()
        arrays = {"base": outputs[0][cols].values}
        for idx, sc in enumerate(self.scenarios_cfg, start=1):
            arrays[sc["id"]] = outputs[idx][cols].values

        return arrays
