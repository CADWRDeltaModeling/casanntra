# base.suisun

**Highlights / Changes vs previous step**

* name: dsm2.schism → base.suisun
* input_prefix: schism_base → schism_suisun
* output_prefix: {output_dir}/dsm2.schism_base_gru2 → {output_dir}/schism_base.suisun_gru2
* load_model_fname: {output_dir}/dsm2_base_gru2 → {output_dir}/dsm2.schism_base_gru2
* save_model_fname: {output_dir}/dsm2.schism_base_gru2 → {output_dir}/schism_base.suisun_gru2
* main_epochs: 100 → 35
* builder_args.transfer_type: direct → contrastive
* builder_args.contrast_weight: None → 0.5
* builder_args.source_data_prefix: None → schism_base
* builder_args.source_input_mask_regex: None → ['schism_base_2021.csv']

**Transfer type**: contrastive
