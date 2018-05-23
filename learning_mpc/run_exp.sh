#!/bin/bash
# fast
python main.py --env_name pendulum --exp_name test -n 2 -ep 1000 -m 10 -sp 500 -r 100 -d 10

# # slow
# python main.py --exp_name slow1 -n 50 -ep 1000

python plot.py data/test*