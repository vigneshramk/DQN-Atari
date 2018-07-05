Script provided to easily run various configurations:
-----------------------------------------------------

1. train_config.sh to train
2. test_config.sh to test

main.py is the interface to different files implmenting different algorithms and environments
traing_config.sh and test_config.sh call main.py

main.py Usage:
--------------

usage: main.py [-h] [--env ENV] [--alg ALG] [--render RENDER] [--train TRAIN]
               [--model MODEL_FILE]

Deep Q Network Argument Parser

optional arguments:
  -h, --help          show this help message and exit
  --env ENV           gym environment name eg. CartPole-v0, MountainCar-v0,
                      SpaceInvaders-v0
  --alg ALG           Choose the algorithm to use between lnQ_NR, lnQ, DQN and
                      DuQn
  --render RENDER     Choose whether to render or not
  --train TRAIN       Choose whether to train or test
  --model MODEL_FILE  Specify the model_file to load and test

Eg. python main.py --env MountainCar-v0 --alg DQN --train 1 --render 1


train_config.sh usage:
----------------------

Eg. traing_config.sh 1

config number | Action
-------------   -------
0             | Trains linear Q without replay for CartPole-v0 with rendering on
1             | Trains linear Q without replay for MountainCar-v0 with rendering on
2             | Trains linear Q with replay for CartPole-v0 with rendering on
3             | Trains linear Q with replay for MountainCar-v0 with rendering on
4             | Trains DQN for CartPole-v0 with rendering on
5             | Trains DQN for MountainCar-v0 with rendering on
6             | Trains DQN for SpaceInvaders-v0 with rendering on
7             | Trains Dueling Q for CartPole-v0 with rendering on
8             | Trains Dueling Q for MountainCar-v0 with rendering on


test_config.sh usage:
----------------------

Eg. test_config.sh 0 model.h5


Takes model file as extra input

config number | Action
-------------   -------
0             | Tests linear Q without replay for CartPole-v0 with rendering on
1             | Tests linear Q without replay for MountainCar-v0 with rendering on
2             | Tests linear Q with replay for CartPole-v0 with rendering on
3             | Tests linear Q with replay for MountainCar-v0 with rendering on
4             | Tests DQN for CartPole-v0 with rendering on
5             | Tests DQN for MountainCar-v0 with rendering on
6             | Tests DQN for SpaceInvaders-v0 with rendering on
7             | Tests Dueling Q for CartPole-v0 with rendering on
8             | Tests Dueling Q for MountainCar-v0 with rendering on


