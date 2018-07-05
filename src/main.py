import DQN_keras as DQN
import lnQ_keras as lnQ
import lnQ_NR_keras as lnQ_NR
import DuelingQ_keras as DuQN
import SI5 as SI5
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, help='gym environment name eg. CartPole-v0, MountainCar-v0, SpaceInvaders-v0')
    parser.add_argument('--alg',dest='alg', type=str,default='DQN', help='Choose the algorithm to use between lnQ_NR, lnQ, DQN and DuQn')
    parser.add_argument('--render',dest='render',type=int,default=0, help='Choose whether to render or not')
    parser.add_argument('--train',dest='train',type=int,default=1, help='Choose whether to train or test')
    parser.add_argument('--model',dest='model_file',type=str, help='Specify the model_file to load and test')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.alg == 'lnQ_NR':
        print('\n\n\nRunning Linear Q Network without replay...\n\n\n')
        lnQ_NR.lnQ_NR_main(args)
    elif args.alg == 'lnQ':
        print('\n\n\nRunning Linear Q Network with replay...\n\n\n')
        lnQ.lnQ_main(args)
    elif args.alg == 'DQN':
        print('\n\n\nRunning DQN with replay...\n\n\n')
        DQN.DQN_main(args)
    elif args.alg == 'DuQN':
        print('\n\n\nRunning Dueling Q Network replay...\n\n\n')
        DuQN.DuQN_main(args)
    elif args.alg == 'SI5':
        print('\n\n\nRunning SI5 Q Network replay...\n\n\n')
        SI5.SI5_main(args)



