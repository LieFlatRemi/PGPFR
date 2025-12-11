#!/bin/bash

# Run trial
trial_id=$1
src_dir=$2
split_type=$3
gpu=$4
datasets=$5
baselines=$6

cd ../src/drivers

run_driver() {

    python main_driver.py \
    --train ${train} \
    --dataset ${dataset} \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${root_dir} \
    --log_dir ${log_dir} \
    --gpu ${gpu} \
    --save_last_only \
    --trial_id ${trial_id}
}


for dataset_name in ${datasets[*]}; do
    if [ $dataset_name = "hgr_shrec_2017" ] 
    then
        dataset="hgr_shrec_2017"
#        root_dir="/home/hdd1/xxx/SHREC_2017"
        root_dir=/mnt/f/yuecheng/code/PytorchProject/GestureRecognition/dataset/SHREC17
    elif [ $dataset_name = "ego_gesture" ]
    then
        dataset="ego_gesture"
        root_dir="/home/hdd1/xxx/ego_gesture_v4"
    fi

    for baseline_name in ${baselines[*]}; do
        ############################ Run baseline ############################
        #Train
        train=1
        cfg_file=/mnt/f/yuecheng/code/PytorchProject/GestureRecognition/PGPFR-3/src/configs/params/$dataset/$baseline_name.yaml
        log_dir=/mnt/f/yuecheng/code/PytorchProject/GestureRecognition/PGPFR-3/output/$dataset/$baseline_name
        trial_id=$trial_id
        run_driver

        #Test
        train=-1
        run_driver
    done
done


