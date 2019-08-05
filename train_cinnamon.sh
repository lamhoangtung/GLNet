export CUDA_VISIBLE_DEVICES=3
python3 train_cinnamon.py \
--n_class 2 \
--data_path "./data/all_prj/" \
--model_path "./experiments/cinnamon/saved_models/" \
--log_path "./experiments/cinnamon/" \
--task_name "fpn_local2global.508_4.28.2019_lr2e5" \
--mode 3 \
--batch_size 1 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_global.508_4.28.2019_lr2e5.pth" \
--path_g2l "fpn_globa2local.508_4.28.2019_lr2e5.pth" \
--path_l2g "fpn_local2global.508_4.28.2019_lr2e5.pth" \
--num_workers 16
# --path_g "cityscapes_global.800_4.5.2019.lr5e5.pth" \
# --path_g2l "fpn_global2local.508_deep.cat.1x_fmreg_ensemble.p3.0.15l2_3.19.2019.lr2e5.pth" \
# --path_l2g "fpn_local2global.508_deep.cat.1x_fmreg_ensemble.p3_3.19.2019.lr2e5.pth" \
