# Validation
CUDA_VISIBLE_DEVICES=1 \
python validate.py \
   --dataset fashioniq \
   --model-s1-path ./models/combiner_trained_on_fiq_ViT-B/16_2023-12-27_01:35:11/saved_models/model.pt \
   --projection-dim 4096 \
   --hidden-dim 8192 \
   --clip-model-name ViT-B/16 \
   --target-ratio 1.25 \
   --transform targetpad \
   --network clip4cir_maple_final_s2 \
   --final \
   --num-workers 64 \
   --maple-n-ctx 3 \
   --maple-ctx-init 'a photo of' \
   --maple-prompt-depth 9 \
   --asynchronous \
   --fixed-image-encoder \
   --combiner combiner_v5 \
   --model-s2-path ./models/combiner_trained_on_fiq_ViT-B/16_2023-12-27_01:35:11/saved_models/combiner.pt \
   --img2txt-model-path ./saved_models/MPAC/fiq_img2txt.pt \
   --optimizer combiner





