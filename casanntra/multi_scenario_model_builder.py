from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import GRU, LSTM, Dense, Reshape, Concatenate
from keras.models import load_model

from casanntra.model_builder import ModelBuilder, UnscaleLayer, ScaledMaskedMAE, ScaledMaskedMSE, masked_mae, masked_mse
from casanntra.multi_stage_model_builder import MultiStageModelBuilder

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
        if tt not in ("direct", "contrastive", "multi-direct"):
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
        return self.transfer_type in ("contrastive", "multi-direct") and len(self.scenarios_cfg) > 0

    def is_multi_scenario_step(self) -> bool:
        return self.transfer_type in ("contrastive", "multi-direct") and len(self.scenarios_cfg) > 0

    def num_outputs(self):
        return len(self._supervised_keys) if self._supervised_keys else 1

    def _contrast_loss_scales(self):
        """Loss scales for the contrast heads (contrast-scale A/B, 2026-08-27).

        When builder_args provides 'contrast_scales' (station -> typical contrast
        magnitude), contrast residuals are normalized by those instead of the
        absolute output scales. When the key is absent, returns the absolute
        output scales - byte-identical to the pre-A/B behavior.
        """
        cs = self.builder_args.get("contrast_scales", None)
        if cs is None:
            return list(self.output_names.values())
        missing = [s for s in self.output_names if s not in cs]
        if missing:
            raise ValueError(f"contrast_scales missing stations: {missing}")
        return [float(cs[s]) for s in self.output_names]

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

        # direct: base only, contrastive/multi-direct: base + scenarios
        if self.transfer_type not in ("contrastive", "multi-direct") or len(self.scenarios_cfg) == 0:
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
            # Only add contrast heads for contrastive mode (not multi-direct)
            if self.transfer_type == "contrastive":
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
            raise ValueError(
                f"Recurrent branch layer '{layer.name}' received 2D input {tensor.shape}. "
                f"Trunk's last layer must have return_sequences=True for recurrent branches."
            )
        return layer(tensor)

    def _build_head(self, feat, head_id: str, use_branch: bool, dense_name: Optional[str] = None) -> Dict[str, tf.Tensor]:
        z = feat
        if use_branch and self.branch_layers:
            for idx, spec in enumerate(self.branch_layers):
                cls = self._layer_cls(spec["type"])
                layer_name = self._branch_layer_name(spec, head_id, idx)
                kw = {k: v for k, v in spec.items() if k not in {"type", "name"}}
                if cls in (GRU, LSTM):
                    kw.setdefault("activation", "sigmoid")
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

        base_model.load_weights(self.load_model_fname + ".weights.h5")
        return base_model

    def _get_old_head_weights(self, prev: Model):
        """Get head weights from previous model, trying various possible layer names."""
        possible_names = ["head_base_scaled", "source_scaled", "target_scaled", "out_scaled", "out_target_scaled"]
        prev_layer_names = {l.name for l in prev.layers}
        for name in possible_names:
            if name in prev_layer_names:
                print(f"[weights] found previous head weights from: {name}")
                return prev.get_layer(name).get_weights()
        raise ValueError(f"loaded model has no head layer among {possible_names}")

    def build_model(self, input_layers, input_data):
        # Load previous model FIRST (like MSTAGE approach)
        prev = self._load_previous_model()

        if prev is not None:
            # REUSE loaded model's computation graph (MSTAGE approach)
            # This preserves all internal layer state exactly
            if isinstance(prev.input, list):
                input_layer = {layer.name: layer for layer in prev.input}
            else:
                input_layer = prev.input

            # Check if recurrent branches need full sequences from trunk
            has_recurrent_branch = (
                self.branch_layers
                and any(self._layer_cls(bl["type"]) in (GRU, LSTM)
                        for bl in self.branch_layers)
                and (self.per_scenario_branch or self.include_source_branch)
            )

            last_trunk_name = self.trunk_spec[-1].get("name", f"trunk_{len(self.trunk_spec)}")
            last_trunk_layer = prev.get_layer(last_trunk_name)

            if has_recurrent_branch and not last_trunk_layer.return_sequences:
                # Rebuild last trunk layer with return_sequences=True so
                # recurrent branches receive the full temporal sequence.
                # GRU weights are independent of return_sequences.
                print(f"[MultiScenario] Rebuilding {last_trunk_name} with return_sequences=True for recurrent branches")

                if len(self.trunk_spec) > 1:
                    prev_trunk_name = self.trunk_spec[-2].get("name", f"trunk_{len(self.trunk_spec)-1}")
                    prev_trunk_output = prev.get_layer(prev_trunk_name).output
                else:
                    prev_trunk_output = prev.get_layer("stacked").output

                last_spec = self.trunk_spec[-1]
                cls = self._layer_cls(last_spec["type"])
                kw = {k: v for k, v in last_spec.items() if k not in {"type", "name", "trainable"}}
                kw["return_sequences"] = True
                if cls in (GRU, LSTM):
                    kw.setdefault("activation", "sigmoid")
                new_last_layer = cls(name=last_trunk_name + "_seq", **kw)
                feat = new_last_layer(prev_trunk_output)

                # Copy weights from loaded model's last trunk layer
                new_last_layer.set_weights(last_trunk_layer.get_weights())
                print(f"[MultiScenario] Copied weights from {last_trunk_name} to {last_trunk_name}_seq")
            else:
                # Standard path: reuse loaded trunk output directly
                feat = last_trunk_layer.output

            # Get old head weights for initializing new heads
            old_head_weights = self._get_old_head_weights(prev)

            print(f"[MultiScenario] Reusing computation graph from loaded model")
            print(f"[MultiScenario] Feature extractor: {last_trunk_name}")
        else:
            # No transfer learning - build fresh (Step 1 only)
            input_layer = input_layers
            prepro = self.prepro_layers(input_layers, input_data)
            expanded = [Reshape((self.ndays, 1))(t) for t in prepro]
            x = Concatenate(axis=-1, name="stacked")(expanded)
            feat = self._build_trunk(x)
            old_head_weights = None

        # Build heads on top of feature extractor
        outputs = {}
        head_tensors: Dict[str, tf.Tensor] = {}

        for spec in self.head_plan:
            if spec["kind"] != "dense":
                continue
            use_branch = spec["builder"] == "branch"
            pack = self._build_head(feat, head_id=spec["head_id"], use_branch=use_branch, dense_name=spec.get("dense_name"))
            outputs[spec["out_name"]] = pack["y_unscaled"]
            head_tensors[spec["head_id"]] = pack["y_unscaled"]

        # Build contrast heads (subtract layers)
        for spec in self.head_plan:
            if spec["kind"] != "contrast":
                continue
            outputs[spec["out_name"]] = layers.Subtract(name=spec["out_name"])([
                head_tensors[spec["pos_head"]],
                head_tensors[spec["neg_head"]],
            ])

        ann = Model(inputs=input_layer, outputs=outputs, name="multi_scenario_model")

        # Initialize head weights from previous model
        if old_head_weights is not None:
            ann.get_layer("head_base_scaled").set_weights(old_head_weights)
            print("[weights] initialized head_base_scaled from previous model")
            if self.init_targets_from_source:
                for sc in self.scenarios_cfg:
                    ann.get_layer(f"head_{sc['id']}_scaled").set_weights(old_head_weights)
                    print(f"[weights] initialized head_{sc['id']}_scaled from previous model")

        # Initialize branch weights from trunk if possible
        if prev is not None and self.branch_layers and self.init_targets_from_source:
            last_trunk_weights = last_trunk_layer.get_weights()
            trunk_shapes = [w.shape for w in last_trunk_weights]
            existing = {l.name for l in ann.layers}
            for idx, spec in enumerate(self.branch_layers):
                names = [spec.get("name", f"branch_{idx+1}")] + [self._branch_layer_name(spec, sc["id"], idx) for sc in self.scenarios_cfg]
                for name in names:
                    if name not in existing:
                        continue
                    layer = ann.get_layer(name)
                    if [w.shape for w in layer.get_weights()] != trunk_shapes:
                        raise ValueError(
                            f"cannot seed branch {name} from trunk layer {last_trunk_name}: weight shapes "
                            f"{[w.shape for w in layer.get_weights()]} vs {trunk_shapes}. A GRU kernel is "
                            f"(input_dim, 3*units): the trunk layer sees the previous layer's width as input, "
                            f"the branch sees the trunk's output width, so the copy only works when those widths "
                            f"and the unit counts are equal (e.g. trunk [32, 32] with a 32-unit branch). Otherwise "
                            f"the branch would start from random weights and the experiment would not test a "
                            f"seeded branch.")
                    layer.set_weights(last_trunk_weights)
                    print(f"[weights] initialized {name} from {last_trunk_name}")

        self._apply_trainable_flags(ann)

        print(ann.summary())
        return ann

    pool_and_align_cases = MultiStageModelBuilder.pool_and_align_cases

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
            # loss_dict[spec["out_name"]] = ScaledMaskedMAE(output_scales)  # pre-A/B: contrast scaled by absolute station scales
            loss_dict[spec["out_name"]] = ScaledMaskedMAE(self._contrast_loss_scales())
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
