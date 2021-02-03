#!/bin/bash

for (( counter = 64; counter <= 256; counter *= 2 )); do
        GPU_MODE=0
        MIXED_PRECISION=0
        POLICY_TYPE=16

        working_time=$(python3 app_resnet50.py --batch_size $counter --gpu_mode $GPU_MODE --mixed_precision $MIXED_PRECISION --policy_type $POLICY_TYPE | grep Working | awk '{print $3}')
        echo -e "$counter\t$GPU_MODE\t$MIXED_PRECISION\t$POLICY_TYPE\t$working_time"

        GPU_MODE=1
        working_time=$(python3 app_resnet50.py --batch_size $counter --gpu_mode $GPU_MODE --mixed_precision $MIXED_PRECISION --policy_type $POLICY_TYPE | grep Working | awk '{print $3}')
        echo -e "$counter\t$GPU_MODE\t$MIXED_PRECISION\t$POLICY_TYPE\t$working_time"

        GPU_MODE=0
        MIXED_PRECISION=1
        POLICY_TYPE=16
        working_time=$(python3 app_resnet50.py --batch_size $counter --gpu_mode $GPU_MODE --mixed_precision $MIXED_PRECISION --policy_type $POLICY_TYPE | grep Working | awk '{print $3}')
        echo -e "$counter\t$GPU_MODE\t$MIXED_PRECISION\t$POLICY_TYPE\t$working_time"

        POLICY_TYPE=32
        working_time=$(python3 app_resnet50.py --batch_size $counter --gpu_mode $GPU_MODE --mixed_precision $MIXED_PRECISION --policy_type $POLICY_TYPE | grep Working | awk '{print $3}')
        echo -e "$counter\t$GPU_MODE\t$MIXED_PRECISION\t$POLICY_TYPE"
        echo $working_time

        GPU_MODE=1
        POLICY_TYPE=16
        working_time=$(python3 app_resnet50.py --batch_size $counter --gpu_mode $GPU_MODE --mixed_precision $MIXED_PRECISION --policy_type $POLICY_TYPE | grep Working | awk '{print $3}')
        echo -e "$counter\t$GPU_MODE\t$MIXED_PRECISION\t$POLICY_TYPE\t$working_time"

        POLICY_TYPE=32
        working_time=$(python3 app_resnet50.py --batch_size $counter --gpu_mode $GPU_MODE --mixed_precision $MIXED_PRECISION --policy_type $POLICY_TYPE | grep Working | awk '{print $3}')
        echo -e "$counter\t$GPU_MODE\t$MIXED_PRECISION\t$POLICY_TYPE\t$working_time"
done

