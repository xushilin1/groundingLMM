deepspeed --master_port 29500 train.py \
    --version MBZUAI/GLaMM-GranD-Pretrained \
    --dataset_dir ./data/ \
    --vision_pretrained /data/home/xsl/sam_vit_h_4b8939.pth \
    --exp_name output/gcg \
    --lora_r 8 --lr 3e-4 \
    --pretrained \
    --use_segm_data \
    --seg_dataset "Semantic_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG" \
    --segm_sample_rates "1,3,3,3,1" \
    --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal" \
    --epochs 10 --steps_per_epoch 500 --mask_validation



# GCG (Grounded Conversation Generation)
# if lora_r == 0 freeze the vision model and llm
# if lora_r > 0 
    # train llm q/v with lora
    # train "lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_encoder"
    # freeze vision model and projector!!!!
deepspeed --master_port 29500 train.py \
    --version MBZUAI/GLaMM-GranD-Pretrained \
    --dataset_dir ./data/ \
    --vision_pretrained /data/home/xsl/sam_vit_h_4b8939.pth \
    --exp_name output/gcg \
    --lora_r 8 --lr 3e-4 --pretrained \
    --use_segm_data \
    --seg_dataset "Semantic_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG" \
    --segm_sample_rates "1,3,3,3,1" \
    --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal" \
    --epochs 10 --steps_per_epoch 500 --mask_validation
