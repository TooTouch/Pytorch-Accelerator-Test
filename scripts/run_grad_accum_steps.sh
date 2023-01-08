python main.py \
--exp-name grad_accum_steps \
--opt-name SGD \
--aug-name strong \
--epochs 200 \
--batch-size 512 \
--grad-accum-steps 4 \
--log-interval 2 \
--use-wandb
