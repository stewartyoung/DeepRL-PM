https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-portfolio-allocation-9b417660c7cd
conda create -n finRL python=3.7.10
conda activate finRL
pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git
pip install jupyter ipykernel
python -m ipykernel install --user --name finRL --display-name "Python (finRL)"
paste contents of https://github.com/AI4Finance-LLC/FinRL-Library/files/5879628/env.zip to /Users/stewart/miniconda3/envs/finRL/lib/python3.7/site-packages/finrl/env/
jupyter notebook
