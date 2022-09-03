for seed in 0 1 2
do
    for env in 0 1 2 3
    do
        CUDA_VISIBLE_DEVICES=0,3 python main.py --test_envs $env --seed $seed;
    done
done