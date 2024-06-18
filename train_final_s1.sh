# CLIP fine-tuning
CUDA_VISIBLE_DEVICES=6 \
python clip_fine_tune.py \
   --dataset fashioniq \
   --experiment-name fiq_clip_vitb16_231229_test \
   --num-epochs 100 \
   --clip-model-name ViT-B/16 \
   --encoder none \
   --learning-rate 2e-6 \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1 \
   --network clip4cir_maple_final_s1 \
   --final \
   --num-workers 64 \
   --fixed-image-encoder \
   --img2txt-model-path ./saved_models/MPAC/fiq_img2txt.pt \
   --asynchronous \
   --clip-model-path ./saved_models/MPAC/fiq_tuned_clip_best.pt \
   --clip-image-encoder-path ./saved_models/MPAC/fiq_image_encoder.pt \
   --maple-prompt-depth 9 \
   --maple-ctx-init 'a photo of' \
   --maple-n-ctx 3


# CUDA_VISIBLE_DEVICES=6 \
# python clip_fine_tune.py \
#    --dataset cirr \
#    --experiment-name cirr_clip_vitb16_240112_test \
#    --num-epochs 100 \
#    --clip-model-name ViT-B/16 \
#    --encoder none \
#    --learning-rate 2e-6 \
#    --batch-size 128 \
#    --transform targetpad \
#    --target-ratio 1.25 \
#    --save-training \
#    --save-best \
#    --validation-frequency 1 \
#    --network clip4cir_maple_final_s1 \
#    --final \
#    --num-workers 8 \
#    --fixed-image-encoder \
#    --img2txt-model-path /amax/home/xtyao/cir/clip4cir_pic2word/models/clip_finetuned_on_cirr_ViT-B/16_2023-12-24_17:42:41/saved_models/best_model.pt \
#    --asynchronous \
#    --clip-model-path /amax/home/xtyao/cir/MMPT/saved_models/cirr_clip_ft_27_125830.pt \
#    --clip-image-encoder-path /amax/home/xtyao/cir/MMPT/saved_models/cirr_image_encoder_avgrecall_27_125830.pt \
#    --maple-prompt-depth 11 \
#    --maple-ctx-init 'a photo of' \
#    --maple-n-ctx 3
   

