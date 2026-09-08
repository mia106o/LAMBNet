#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/tiny/segman_t_sos_palsar.py \
2 \
--work-dir outputs/sos/palsar/segman_t \
--seed 0
