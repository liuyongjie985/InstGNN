base_dir=/home/liuyongjie/InstGNN/data/sogou_collect
out_dir=/home/liuyongjie/InstGNN/data/sogou_tencent/
train_dir="$out_dir"train/
valid_dir="$out_dir"valid/
#rm -rvf "$out_dir"TS_JSON 
#rm -rvf "$out_dir"TS_PIC/ 
#rm -rvf "$out_dir"node_feature_json/ 
#rm -rvf "$out_dir"node_label_json/ 
#rm -rvf "$out_dir"edge_feature_json/ 
#rm -rvf "$out_dir"edge_label_json/ 
#rm -rvf "$out_dir"id2originid_json
#rm -rvf "$out_dir"train
#rm -rvf "$out_dir"valid

#mkdir "$out_dir"edge_feature_json
#mkdir "$out_dir"edge_label_json
#mkdir "$out_dir"id2originid_json
#mkdir "$out_dir"node_feature_json
#mkdir "$out_dir"node_label_json
#mkdir "$out_dir"TS_JSON
#mkdir "$out_dir"TS_PIC
#mkdir "$out_dir"train
#mkdir "$out_dir"valid

#python dealInkml.py $base_dir "$out_dir"TS_JSON "$out_dir"TS_PIC/ "$out_dir"node_feature_json/ "$out_dir"node_label_json/ "$out_dir"edge_feature_json/ "$out_dir"edge_label_json/ "$out_dir"id2originid_json TS

rm -rvf "$out_dir"train
rm -rvf "$out_dir"valid
mkdir "$out_dir"train
mkdir "$out_dir"valid
mkdir "$train_dir"edge_feature_json
mkdir "$train_dir"edge_label_json
mkdir "$train_dir"id2originid_json
mkdir "$train_dir"node_feature_json
mkdir "$train_dir"node_label_json
mkdir "$train_dir"TS_JSON
mkdir "$train_dir"TS_PIC
mkdir "$valid_dir"edge_feature_json
mkdir "$valid_dir"edge_label_json
mkdir "$valid_dir"id2originid_json
mkdir "$valid_dir"node_feature_json
mkdir "$valid_dir"node_label_json
mkdir "$valid_dir"TS_JSON
mkdir "$valid_dir"TS_PIC

python splitTrainVal.py --base_path $out_dir --valid_num 100
