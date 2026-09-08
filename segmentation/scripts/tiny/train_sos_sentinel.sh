#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/tiny/segman_t_sos_sentinel.py \
2 \
--work-dir outputs/sos/sentinel/segman_t \
--seed 0
