# standard packages
from utils.pickling import pickleIt
from utils.plotting import plotIt
from utils.metrics import SharpeRatio, MDD
from universal import algos
import seaborn as sns
from DeepRL.utils.misc import run_episodes
import shutil
import pickle
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
from networks.base_networks import DeterministicActorNet, DeterministicCriticNet
from DeepRL.component import GaussianPolicy, HighDimActionReplay, OrnsteinUhlenbeckProcess
from DeepRL.agent import ProximalPolicyOptimization, DisjointActorCriticNet
import gym
from DeepRL.utils import Logger
from agents.DDPGAgent import DDPGAgent
from environment.config import Config
from wrappers import TransposeHistory, ConcatStates, SoftmaxActions, DeepRLWrapper
from environment.stock_environment import PortfolioEnv
import argparse
from utils.dataPreprocessing import preProcessData
import datetime
import logging
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
# us a non interactive plotting environment for this case
mpl.use('TkAgg')
plt.style.use('ggplot')
# logging
logger = log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()
log.info('%s logger started.', __name__)
# models and tensorboard logging
# save dir
ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
save_path = './outputs/pytorch-DDPG/pytorch-DDPG-EIIE-action-crypto-%s.model' % ts
save_path
try:
    os.makedirs(os.path.dirname(save_path))
except OSError:
    pass

os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.abspath('..'))
os.sys.path.append(os.path.abspath('DeepRL'))
gym.logger.setLevel(logging.INFO)
# pytorch
# load
# train agent
# view agent rewards
# universal algorithms
# backtesting metrics
# plot the results and save to file
# pickle the results


def save_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = 'data/%s-%s-model-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    agent.save(save_file)
    print(save_file)


def load_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = 'data/%s-%s-model-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    new_states = pickle.load(open(save_file, 'rb'))
    states = agent.worker_network.load_state_dict(new_states)


def load_stats_ddpg(agent):
    agent_type = agent.__class__.__name__
    online_stats_file = 'data/%s-%s-online-stats-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    try:
        steps, rewards = pickle.load(open(online_stats_file, 'rb'))
    except FileNotFoundError:
        steps = []
        rewards = []
    df_online = pd.DataFrame(
        np.array([steps, rewards]).T, columns=['steps', 'rewards'])
    if len(df_online):
        df_online['step'] = df_online['steps'].cumsum()
        df_online.index.name = 'episodes'

    stats_file = 'data/%s-%s-all-stats-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    try:
        stats = pickle.load(open(stats_file, 'rb'))
    except FileNotFoundError:
        stats = {}
    df = pd.DataFrame(stats["test_rewards"], columns=['rewards'])
    if len(df):
        #         df["steps"]=range(len(df))*50

        df.index.name = 'episodes'
    return df_online, df


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # e.g. --data = "CAC40"
    parser.add_argument('--data', type=str, required=True,
                        help='Market Index Dataset')
    # e.g. --proportion_assets 1.0 0.5
    parser.add_argument('--proportion_assets', nargs="+", type=float,
                        required=True, help='Proportion of assets to simulate with')
    # e.g. --num_repeats 4
    parser.add_argument('--num_repeats', type=int, required=True,
                        help='How many times to repeat each combination of hyperparameters')
    # e.g. --gpu True
    parser.add_argument('--gpu', type=bool, required=True,
                        help="Use the gpu or not")
    args = parser.parse_args()
    return args


def universalPortfolioStrat(env, algo, seed=0):
    env.seed(0)
    np.random.seed(0)
    # start the environment from the start using reset()
    state = env.reset()
    # unwrapped removes all wrappers the environment instance has,
    # then step through the environment with the for loop
    for _ in range(env.unwrapped.sim.steps):
        history = pd.DataFrame(
            state[0, :, :], columns=env.unwrapped.src.asset_names)
        # modern portfolio theory approach to universal portfolios needs cash as 1st column
        history["CASH"] = 1
        history = history[['CASH'] + env.unwrapped.src.asset_names]
        # some strategies need a history - history (BSF, OLMAR),
        # others just need the previous time step - x (ONS, EG, RMR)
        x = history.iloc[-1]
        # portfolio weights w0 from the previous time step
        last_weights = env.unwrapped.sim.w0
        # fill algo object with stock returns history
        algo.init_step(history)
        # some strategies don't require history
        try:
            action = algo.step(x, last_weights, history)
        except TypeError:
            action = algo.step(x, last_weights)
        action = getattr(action, 'value', action)
        # format for universal portfolio theory strategy
        if isinstance(action, np.matrixlib.defmatrix.matrix):
            action = np.array(action.tolist()).T[0]
        # take the action on the environment, observing new state and reward
        # done=at last time step (True), o.w. (False), info=debugging dictionary
        state, reward, done, info = env.step(action)
        # if at last time step, break out of for loop
        if done:
            break
    # make dataframe of the returns information
    df = pd.DataFrame(env.unwrapped.infos)
    df.index = pd.to_datetime(df['date']*1e9)

    return df['portfolio_value'], df


if __name__ == "__main__":
    # parse args
    args = parse_args()
    # setup tensorboard logging
    from tensorboard_logger import configure, log_value
    tag = 'ddpg-' + args.data
    print('tensorboard --logdir '+"runs/" + args.data)
    try:
        configure("runs/" + tag)
    except ValueError as e:
        print(e)
        pass
    # import data
    problemsAll = {"CAC40": [], "DAX": ["VONOVIA", "COVESTRO", "MTU AERO ENGINES HLDG.", "LINDE (FRA)"],
                   "FTSE100": ["GLENCORE", "EXPERIAN", "INTL.CONS.AIRL.GP.", "COCA-COLA HBC", "HARGREAVES LANSDOWN", "MONDI",
                               "OCADO GROUP", "STANDARD LIFE ABERDEEN", "AUTO TRADER GROUP", "EVRAZ", "NMC HEALTH", "RIGHTMOVE",
                               "HIKMA PHARMACEUTICALS", "PHOENIX GROUP HDG.", "POLYMETAL INTERNATIONAL", "TUI (LON)",
                               "ROYAL DUTCH SHELL A(LON)", "SMURFIT KAPPA GP. (LON)"],
                   "NIKKEI225": ["RECRUIT HOLDINGS", "JAPAN POST HOLDINGS", "OTSUKA HOLDINGS", "DAI-ICHI LIFE HOLDINGS",
                                 "SOMPO HOLDINGS", "INPEX", "JXTG HOLDINGS", "IDEMITSU KOSAN", "MEIJI HOLDINGS", "MITSUBISHI CHM.HDG.",
                                 "SONY FINANCIAL HOLDINGS", "AOZORA BANK", "CONCORDIA FINANCIAL GP.", "DENA", "NIPPON PAPER INDUSTRIES",
                                 "SUMCO", "TOKYU FUDOSAN HOLDINGS"],
                   "SP500": ["FACEBOOK CLASS A", "VISA 'A'", "MASTERCARD", "ABBVIE", "PAYPAL HOLDINGS", "PHILIP MORRIS INTL.",
                             "BROADCOM", "CHARTER COMMS.CL.A", "T-MOBILE US", "ZOETIS A", "GENERAL MOTORS", "INGERSOLL RAND",
                             "SERVICENOW", "MSCI", "INTERCONTINENTAL EX.", "KINDER MORGAN", "HCA HEALTHCARE", "LAS VEGAS SANDS",
                             "DELTA AIR LINES", "DOLLAR GENERAL", "KRAFT HEINZ", "MARATHON PETROLEUM", "PHILLIPS 66",
                             "DISCOVER FINANCIAL SVS.", "IHS MARKIT", "TRANSDIGM GROUP", "TWITTER", "VERISK ANALYTICS CL.A",
                             "AMERICAN WATER WORKS", "DIGITAL REALTY TST.", "FIRST REPUBLIC BANK", "FLEETCOR TECHNOLOGIES",
                             "HILTON WORLDWIDE HDG.", "IQVIA HOLDINGS", "TE CONNECTIVITY", "LYONDELLBASELL INDS.CL.A",
                             "UNITED AIRLINES HOLDINGS", "APTIV", "CHIPOTLE MEXN.GRILL", "FORTIVE", "AMERICAN AIRLINES GROUP",
                             "AMERIPRISE FINL.", "EXPEDIA GROUP", "HEWLETT PACKARD ENTER.", "KEYSIGHT TECHNOLOGIES",
                             "SYNCHRONY FINANCIAL", "ARISTA NETWORKS", "CDW", "CF INDUSTRIES HDG.", "CITIZENS FINANCIAL GROUP",
                             "FORTINET", "LIVE NATION ENTM.", "PAYCOM SOFTWARE", "ULTA BEAUTY", "XYLEM", "ALLEGION",
                             "BROADRIDGE FINL.SLTN.", "CAPRI HOLDINGS", "CELANESE", "CONCHO RESOURCES", "LAMB WESTON HOLDINGS",
                             "LEIDOS HOLDINGS", "CBOE GLOBAL MARKETS", "COTY CL.A", "DIAMONDBACK ENERGY", "INVESCO",
                             "MARKETAXESS HOLDINGS", "NIELSEN", "WESTERN UNION", "WESTROCK", "DISCOVERY SERIES A",
                             "FORTUNE BNS.HM.& SCTY.", "HANESBRANDS", "HNTGTN.INGALLS INDS.", "NORWEGIAN CRUISE LINE HDG.",
                             "UNDER ARMOUR A", "IPG PHOTONICS", "NEWS 'A'", "ALPHABET 'C'", "DISCOVERY SERIES C", "NEWS 'B'",
                             "UNDER ARMOUR 'C'", ],
                   "TSX": ["CENOVUS ENERGY", "GIBSON ENERGY", "MEG ENERGY", "PAREX RESOURCES", "PRAIRIESKY ROYALTY",
                           "SEVEN GENERATIONS ENERGY", "TOURMALINE OIL", "ENERFLEX WNI.", "SECURE ENERGY SERVICES",
                           "WHITECAP RESOURCES"]}
    problems = problemsAll[args.data]
    keep = []
    for proportion in args.proportion_assets:
        data = preProcessData("../../data/" + args.data + ".csv",
                              removeStocks=problems, keepOnly=keep, proportionAssets=proportion)
        train = int(data.shape[0]*0.8)
        data_train = data.iloc[:train, :]
        data_test = data.iloc[train:, :]
        for repeat in range(args.num_repeats):
            print("\ndata:", args.data, "gpu:", args.gpu, "proportion_assets:", proportion,
                  "repeat_no:", repeat, "\n")
            # instantiate environments

            def task_fn():
                env = PortfolioEnv(df=data_train, steps=2868, output_mode='EIIE', filename=args.data+"train",
                                   timestamp=ts, proportion=proportion, repeat=repeat)
                env = TransposeHistory(env)
                env = ConcatStates(env)
                env = SoftmaxActions(env)
                env = DeepRLWrapper(env)
                return env

            def task_fn_test():
                env = PortfolioEnv(df=data_test, steps=668, output_mode='EIIE', filename=args.data+"test",
                                   timestamp=ts, proportion=proportion, repeat=repeat)
                env = TransposeHistory(env)
                env = ConcatStates(env)
                env = SoftmaxActions(env)
                env = DeepRLWrapper(env)
                return env

            task = task_fn()
            # configure agent
            config = Config()
            config.task_fn = task_fn
            task = config.task_fn()
            config.actor_network_fn = lambda: DeterministicActorNet(
                task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu, batch_norm=False, gpu=args.gpu)
            config.critic_network_fn = lambda: DeterministicCriticNet(
                task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False, gpu=args.gpu)
            config.network_fn = lambda: DisjointActorCriticNet(
                config.actor_network_fn, config.critic_network_fn)
            config.actor_optimizer_fn = lambda params: torch.optim.Adam(
                params, lr=4e-5)
            config.critic_optimizer_fn = lambda params: torch.optim.Adam(
                params, lr=5e-4, weight_decay=0.001)
            config.replay_fn = lambda: HighDimActionReplay(
                memory_size=600, batch_size=64)
            config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
                size=task.action_dim, theta=0.15, sigma=0.2, sigma_min=0.00002, n_steps_annealing=10000)
            config.discount = 0.0
            config.min_memory_size = 50
            config.target_network_mix = 0.001
            config.max_steps = 150000
            config.max_episode_length = 3000
            config.target_network_mix = 0.01
            config.noise_decay_interval = 100000
            config.gradient_clip = 20
            config.min_epsilon = 0.1
            config.reward_scaling = 1000
            config.test_interval = 10
            config.test_repetitions = 1
            config.save_interval = 40
            config.logger = Logger('./log', gym.logger)
            config.tag = tag
            agent = DDPGAgent(config)
            # train agent
            start = time.time()
            agent.task._plot = agent.task._plot2 = None
            try:
                run_episodes(agent)
                end = time.time()
                print("Minutes to train:", (end-start)/60)
            except KeyboardInterrupt as e:
                save_ddpg(agent)
                end = time.time()
                print("Minutes to train:", (end-start)/60)
                raise(e)
            # view agent rewards and training
            plt.figure()
            df_online, df = load_stats_ddpg(agent)
            snsplot = sns.regplot(
                x="step", y="rewards", data=df_online, order=1)
            snsplot.figure.savefig("plots/" +
                                   ts + "_" +
                                   args.data+"_" +
                                   "rewards"+"_prop" +
                                   str(proportion) +
                                   "_repeat"+str(repeat)+".png"
                                   )
            portfolio_return = (1+df_online.rewards[-100:].mean())
            returns = task.unwrapped.src.data[0, :, :1]
            market_return = (1+returns).mean()
            # configure test environment
            task = task_fn_test()
            test_steps = 5000
            env_test = task_fn_test()
            agent.task = env_test
            agent.config.max_episode_length = test_steps
            agent.task.reset()
            np.random.seed(0)
            # run in deterministic mode, no training, no exploration
            agent.episode(True)
            agent.task.render('notebook')
            agent.task.render('notebook', True)
            df = pd.DataFrame(agent.task.unwrapped.infos)
            df.index = pd.to_datetime(df['date']*1e9)
            # test universal strategies
            env = task.unwrapped
            price_cols = [
                col for col in df.columns if col.startswith('price')]
            for col in price_cols:
                df[col] = df[col].cumprod()
            df = df[price_cols + ['portfolio_value']]
            universalResults = pd.DataFrame()
            algo_dict = dict(
                BestSoFar=algos.BestSoFar(
                    cov_window=env_test.unwrapped.src.window_length-1),
                RMR=algos.RMR(
                    window=env_test.unwrapped.src.window_length-1, eps=10),
                ONS=algos.ONS(delta=0.2, beta=0.8, eta=0.2),
                UCRP=algos.CRP()
            )
            for name, algo in algo_dict.items():
                perf, info = universalPortfolioStrat(env_test, algo)
                universalResults[name] = perf
            Results = universalResults.join(df["portfolio_value"])
            Results.name = "Results"
            Backtest = dict()
            for strategy in Results.columns:
                Backtest[strategy] = dict(MaximumDrawdown=MDD(Results[strategy]),
                                          SharpeRatio=SharpeRatio(
                    Results[strategy], freq=252),
                    PortfolioValue=Results[strategy].iloc[-1])
            Backtest = pd.DataFrame.from_dict(Backtest)
            plotIt(Results, filename=args.data, timestamp=ts,
                   proportion=proportion, repeat=repeat)
            pickleIt(filename=args.data, timestamp=ts, Results=Results, Backtest=Backtest,
                     proportion=proportion, repeat=repeat)
