collection=$1
visual_feature=$2
model_dir=$3 # only file name
root_path=$4

# training
python method/eval.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir