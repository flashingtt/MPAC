# Combiner training
CUDA_VISIBLE_DEVICES=5 \
python combiner_train.py \
   --dataset fashioniq \
   --experiment-name fiq_comb_ViT-B16_fullft_231230_test \
   --projection-dim 4096 \
   --hidden-dim 8192 \
   --num-epochs 300 \
   --clip-model-name ViT-B/16 \
   --combiner-lr 2e-5 \
   --batch-size 4096 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1 \
   --network clip4cir_maple_final_s2 \
   --combiner combiner_v5 \
   --final \
   --num-workers 64 \
   --fixed-image-encoder \
   --img2txt-model-path ./saved_models/MPAC/fiq_img2txt.pt \
   --model-s1-path ./models/clip_finetuned_on_fiq_ViT-B/16_2023-12-29_11:43:23/saved_models/best_model.pt \
   --asynchronous \
   --optimizer combiner \
   --mu 0.1 \
   --router \
   --maple-prompt-depth 1 \
   --maple-ctx-init 'a photo of' \
   --maple-n-ctx 3

   # --aligner \
   # --txt2img \
   # --clip-model-path /amax/home/xtyao/cir/clip4cir_official/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-05_20:28:16/saved_models/tuned_clip_best.pt \
   
   # --cross-attn-layer 4 \
   # --cross-attn-head 2 \
   # --bsc-loss 
   # fiq /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-18_00:44:19/saved_models/best_model.pt
   # cirr /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_cirr_ViT-B/16_2023-12-21_22:56:38/saved_models/tuned_clip_best.pt

# CUDA_VISIBLE_DEVICES=6 \
# python combiner_train.py \
#    --dataset cirr \
#    --experiment-name cirr_comb_ViT-B16_fullft_2401115_o \
#    --projection-dim 4096 \
#    --hidden-dim 8192 \
#    --num-epochs 300 \
#    --clip-model-name ViT-B/16 \
#    --combiner-lr 2e-5 \
#    --batch-size 4096 \
#    --clip-bs 32 \
#    --transform targetpad \
#    --target-ratio 1.25 \
#    --save-training \
#    --save-best \
#    --validation-frequency 1 \
#    --network clip4cir_maple_final_s2 \
#    --combiner combiner_v5 \
#    --final \
#    --num-workers 32 \
#    --fixed-image-encoder \
#    --img2txt-model-path /amax/home/xtyao/cir/MMPT/saved_models/cirr_pic2word.pt \
#    --model-s1-path /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_cirr_ViT-B/16_2024-01-12_23:27:59/saved_models/tuned_clip_best.pt \
#    --asynchronous \
#    --optimizer combiner \
#    --mu 0.1 \
#    --router \
#    --maple-prompt-depth 11 \
#    --maple-ctx-init 'a photo of' \
#    --maple-n-ctx 3

# CUDA_VISIBLE_DEVICES=2 \
# python combiner_train.py \
#    --dataset fashioniq \
#    --experiment-name fiq_comb_ViT-B16_fullft_231226_test \
#    --projection-dim 4096 \
#    --hidden-dim 8192 \
#    --num-epochs 300 \
#    --clip-model-name ViT-B/16 \
#    --combiner-lr 2e-5 \
#    --batch-size 4096 \
#    --clip-bs 32 \
#    --transform targetpad \
#    --target-ratio 1.25 \
#    --save-training \
#    --save-best \
#    --validation-frequency 1 \
#    --network clip4cir_maple_final_s2 \
#    --combiner combiner_v4 \
#    --final \
#    --num-workers 32 \
#    --fixed-image-encoder \
#    --img2txt-model-path /amax/home/xtyao/cir/clip4cir_pic2word/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-12_16:55:16/saved_models/best_model.pt \
#    --model-s1-path /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-26_11:41:48/saved_models/best_model.pt \
#    --asynchronous \
#    --optimizer combiner \
#    --txt2img \

   # /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-25_15:33:31/saved_models/best_model.pt \






   