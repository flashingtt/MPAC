# Combiner training
CUDA_VISIBLE_DEVICES=4 \
python one_stage_train.py \
   --dataset fashioniq \
   --experiment-name fiq_comb_ViT-B16_fullft_231226_test \
   --projection-dim 4096 \
   --hidden-dim 8192 \
   --num-epochs 300 \
   --clip-model-name ViT-B/16 \
   --combiner-lr 2e-5 \
   --batch-size  \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1 \
   --network clip4cir_maple_final \
   --combiner sum \
   --final \
   --num-workers 32 \
   --fixed-image-encoder \
   --img2txt-model-path /amax/home/xtyao/cir/clip4cir_pic2word/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-12_16:55:16/saved_models/best_model.pt \
   --optimizer combiner_prompt_learner \
   --asynchronous \
   # --txt2img \
   # --model-s1-path /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-18_00:44:19/saved_models/best_model.pt \
   # --clip-model-path /amax/home/xtyao/cir/clip4cir_official/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-05_20:28:16/saved_models/tuned_clip_best.pt \
   # --aligner \
   # --cross-attn-layer 4 \
   # --cross-attn-head 2 \
   # --bsc-loss 
   # fiq /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-18_00:44:19/saved_models/best_model.pt
   # cirr /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_cirr_ViT-B/16_2023-12-21_22:56:38/saved_models/tuned_clip_best.pt

# CUDA_VISIBLE_DEVICES=1 \
# python combiner_train.py \
#    --dataset fashioniq \
#    --experiment-name fiq_comb_ViT-B16_fullft_231225_test \
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
#    --img2txt-model-path /amax/home/xtyao/cir/clip4cir_pic2word/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-12_16:55:16/saved_models/best_model.pt \
#    --model-s1-path /amax/home/xtyao/cir/MMPT/models/clip_finetuned_on_fiq_ViT-B/16_2023-12-25_15:33:31/saved_models/best_model.pt \
#    --asynchronous \
#    --optimizer combiner \
#    --txt2img \








   