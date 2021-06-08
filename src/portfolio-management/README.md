This project relies heavily on a conda environment, with specific versions of packages, and python 3.5.6 in particular. These are some startup instructions when in the root of the project directory.

1. conda create -n rl_pm python=3.5.6
2. conda activate rl_pm
3. conda install pip=20.3.3 (pip >= 21 doesn't support python 3.5)
4. pip install -r requirements.txt
5. you may need to tinker with matplotlib settings so you don't have to render each and every plot, e.g. mpl.use('Agg'), or mpl.use('TkAgg')
6. replace all instances of "scipy.misc" with "scipy.special" in source code
7. mkdir data; mv CAC40.csv data/CAC40.csv
8. cd diss; mkdir data log evaluation_log
9. python GPU-DDPG.py --data CAC40 --proportion_assets 1.0 0.5 --num_repeats 4 --esg_epsilon 0.0 0.001 0.01 0.05 --gpu False
