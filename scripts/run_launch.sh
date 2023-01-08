accelerate launch --fp16 main.py \
--exp-name default \
--opt-name SGD \
--aug-name strong \
--epochs 200 \
--batch-size 512
