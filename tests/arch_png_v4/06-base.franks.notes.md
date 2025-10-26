# base.franks

**Highlights / Changes vs previous step**

* name: base.cache → base.franks
* input_prefix: schism_cache → schism_franks
* output_prefix: {output_dir}/schism_base.cache_gru2 → {output_dir}/schism_base.franks_gru2
* save_model_fname: {output_dir}/schism_base.cache_gru2 → {output_dir}/schism_base.franks_gru2
* builder_args.feature_layers: [{'type': 'GRU', 'units': 32, 'name': 'lay1', 'trainable': True}, {'type': 'GRU', 'units': 16, 'name': 'lay2', 'trainable': True}] → [{'type': 'GRU', 'units': 32, 'name': 'feature1', 'trainable': True}, {'type': 'GRU', 'units': 16, 'name': 'feature2', 'trainable': True}]

**Transfer type**: contrastive
