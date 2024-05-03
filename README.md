# TrojDRL: Evaluation of Backdoor Attacks on Deep Reinforcement Learning

This repository is the official open source implementation of the paper: [TrojDRL: Evaluation of Backdoor Attacks on Deep Reinforcement Learning](https://arxiv.org/pdf/1903.06638.pdf) accepted at DAC 2020.

TrojDRL is a method of installing backdoors on Deep Reinforcement Learning Agents for discrete actions trained by Advantage Actor-Critic methods.

### Installation
- The implementation is based on the [```paac```](https://github.com/Alfredvc/paac) (Parallel Advantage Actor-Critic) method from the [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1705.04862.pdf) that uses Tensorflow 1.13.1.
-  We recommend installing the dependencies using the env.yml 
	- Install [anaconda](https://docs.anaconda.com/anaconda/install/)
	- Open [env.yml](https://github.com/pkiourti/rl_backdoor/blob/master/env.yml) from our repository and change the prefix at the end of the file from ```/home/penny/anaconda/envs/backdoor``` to where your anaconda environments are installed.
	- Run ```conda env create -f env.yml```

### Run
- train: 
```$ python3 train.py --game=breakout --debugging_folder=data/strong_targeted/breakout/ --poison --color=100 --attack_method=targeted --pixels_to_poison_h=3 --pixels_to_poison_v=3 --start_position="0,0" --action=2 --budget=20000 --when_to_poison=uniformly```

- test without attack:
```$ python3 test.py --folder=pretrained/trojaned_models/strong_targeted/breakout_3x3/ --no-poison --index=80000000 --gif_name=breakout```

- test with attack:
```$ python3 test.py --poison --poison_some=200 --color=100 -f=pretrained/trojaned_models/strong_targeted/breakout_3x3 --index=80000000 --gif_name=breakout_attacked```

### Results
- breakout: The target action is move to the right. The trigger is a gray square on the top left.
    <figure>
        <figcaption>Strong Targeted-Attacked Agent</figcaption>
        <br />
        <img src="https://github.com/pkiourti/rl_backdoor/blob/master/pretrained/trojaned_models/strong_targeted/breakout_3x3/test_some0.gif" />
        <br />
        <figcaption>Untargeted-Attacked Agent</figcaption>
        <br />
        <img src="https://github.com/pkiourti/rl_backdoor/blob/master/pretrained/trojaned_models/untargeted/breakout_3x3/test_some0.gif" />
    </figure>

- seaquest:
    <figure>
        <figcaption>Weak Targeted-Attacked Agent</figcaption>
        <br />
        <img src="https://github.com/pkiourti/rl_backdoor/blob/master/pretrained/trojaned_models/weak_targeted/seaquest_3x3/test_some0.gif" />
    </figure>

- (More results under pretrained_models)

### Run

A defense mechanism has been added to the program. It relies on "clean samples" which have to be generated and stored with the following command:

- $ python3 test.py --folder=pretrained/trojaned_models/strong_targeted/breakout_3x3/ --no-poison --index=80000000 --gif_name=breakout --store=True --store_name=no_poison

This will store state data in the pretrained/trojaned_models/strong_targeted/breakout_3x3/state_action_data/no_poison folder. Afterwards the defense mechanism can be run with:

- $ python3 test.py --poison --poison_some=2000 --color=100 -f=pretrained/trojaned_models/strong_targeted/breakout_3x3 --index=80000000 --gif_name=breakout_attacked_san --sanitize=True

**Attention**: The defense mechanism relies on a large number of clean samples in order to work well (in total around 25000). As a SVD is performed on a matrix consisting of all clean sample states, a lot of computing power is required for the execution (it did not work for me) 