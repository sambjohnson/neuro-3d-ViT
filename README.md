# neuro-3d-ViT
Traceback (most recent call last):
  File "/home/b-parker/Desktop/neuro-3d-ViT/neuro-3d-ViT.py", line 451, in <module>
    main()
  File "/home/b-parker/Desktop/neuro-3d-ViT/neuro-3d-ViT.py", line 431, in main
    ) = trainer(
  File "/home/b-parker/Desktop/neuro-3d-ViT/neuro-3d-ViT.py", line 289, in trainer
    val_acc = val_epoch(
  File "/home/b-parker/Desktop/neuro-3d-ViT/neuro-3d-ViT.py", line 216, in val_epoch
    logits = model_inferer(data)
  File "/home/b-parker/miniconda3/envs/neuro-3d-ViT/lib/python3.10/site-packages/monai/inferers/utils.py", line 161, in sliding_window_inference
    roi_size = fall_back_tuple(roi_size, image_size_)
  File "/home/b-parker/miniconda3/envs/neuro-3d-ViT/lib/python3.10/site-packages/monai/utils/misc.py", line 279, in fall_back_tuple
    user = ensure_tuple_rep(user_provided, ndim)
  File "/home/b-parker/miniconda3/envs/neuro-3d-ViT/lib/python3.10/site-packages/monai/utils/misc.py", line 205, in ensure_tuple_rep
    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")
ValueError: Sequence must have length 2, got 3.a