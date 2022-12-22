collection=charades
visual_feature=i3d_rgb_lgi

#CF ratio
clip_scale_w=0.4
frame_scale_w=0.6
# eval
eval_context_bsz=100


exp_id=$1
root_path=$2
device_ids=$3

# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --eval_context_bsz $eval_context_bsz --device_ids $device_ids