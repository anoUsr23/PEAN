collection=tvr
visual_feature=i3d_resnet
q_feat_size=768
margin=0.1
intra_margin=0.1


# eval
eval_context_bsz=40

# dynamic
exp_id=$1
root_path=$2
device_ids=$3

# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --q_feat_size $q_feat_size --margin $margin --device_ids $device_ids \
                    --eval_context_bsz $eval_context_bsz \