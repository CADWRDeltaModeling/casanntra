# base.multi

**Highlights / Changes vs previous step**

* name: dsm2.schism → base.multi
* input_mask_regex: None → ['schism_base_2021\\.csv']
* output_prefix: {output_dir}/dsm2.schism_base_gru2 → {output_dir}/schism_base.multi_gru2
* load_model_fname: {output_dir}/dsm2_base_gru2 → {output_dir}/dsm2.schism_base_gru2
* save_model_fname: {output_dir}/dsm2.schism_base_gru2 → {output_dir}/schism_base.multi_gru2
* init_train_rate: 0.003 → 0.001
* main_train_rate: 0.001 → 0.0005
* main_epochs: 100 → 35
* builder_args.transfer_type: direct → contrastive
* builder_args.contrast_weight: None → 0.5
* builder_args.base_layers: [{'type': 'GRU', 'units': 16, 'name': 'feature1', 'return_sequences': True, 'trainable': True}, {'type': 'GRU', 'units': 16, 'name': 'feature2', 'return_sequences': True, 'trainable': True}] → None
* builder_args.trunk_layers: None → [{'type': 'GRU', 'units': 16, 'name': 'feature1', 'return_sequences': True, 'trainable': True}, {'type': 'GRU', 'units': 16, 'name': 'feature2', 'return_sequences': True, 'trainable': True}]
* builder_args.source_data_prefix: None → schism_base
* builder_args.source_input_mask_regex: None → ['schism_base_2021\\.csv']
* builder_args.scenarios: None → [{'id': 'suisun', 'input_prefix': 'schism_suisun', 'input_mask_regex': ['schism_suisun_2021\\.csv']}, {'id': 'slr', 'input_prefix': 'schism_slr', 'input_mask_regex': ['schism_slr_2021\\.csv']}, {'id': 'cache', 'input_prefix': 'schism_cache', 'input_mask_regex': ['schism_cache_2021\\.csv']}, {'id': 'franks', 'input_prefix': 'schism_franks', 'input_mask_regex': ['schism_franks_2021\\.csv']}]

**Transfer type**: contrastive
