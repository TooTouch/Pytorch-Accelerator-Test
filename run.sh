run_list='default ckp fp16 bf16 grad_accum_steps'

for r in $run_list
do
    bash scripts/run_$r.sh
done
