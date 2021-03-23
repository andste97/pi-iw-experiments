# Hierarchical IW
Implementation of the paper [Hierarchical Width-Based Planning and Learning](https://arxiv.org/abs/2101.06177), appearing in the Proceedings of the 31st International Conference on Automated Planning and Scheduling (ICAPS 2021).

## Summary
In this paper, we present a hierarchical approach to width-based planning.
Based on two sets of high- and low-level features, we partition the state-space into high- and low-level states, where each high-level state contains a subset of low-level states.
Our simple approach to hierarchical planning generates a new high-level node, each time, by performing a low-level search until a state with different high-level features is found (i.e. a state that belongs to another high-level state).
We use this approach to apply width-based planners at two levels of abstraction, and show that the width of a problem can be reduced when choosing the appropriate high-level features. We present experiments in two settings:

* **Classical planning**: We incrementally discover high-level feature candidates and show that our hierarchical approach HIW(1,1) (i.e., using width 1 at both levels of abstraction) outperforms IW(2) in several domains.
* **Pixel-based environments**: We extend [pi-IW](https://github.com/aig-upf/pi-IW) with our hierarchical approach, producing pi-HIW, that learns a policy and a value function from the hierarchical plan, and uses them to guide the low-level search. We use a downsampling of the image as high-level features, and show a big improvement compared to the baseline in spare reward Atari games, where an agent moves in a fixed background (e.g. in Montezuma's Revenge).

## Experiments
The experiments of the paper can be reproduced with scripts [incremental_HIW.py](incremental_HIW.py) and [pi_HIW.py](pi_HIW.py), for the classical planning and pixel-based environments, respectively. To illustrate intermediate steps on these two scripts, we provide [planning_step.py](planning_step.py) where only one planning step is performed, and the resulting high-level feature candidates can be observed, and [online_hierarchical_planning.py](online_hierarchical_planning.py), where we perform on-line replanning with our hierarchical approach without learning. The scripts can be run with default parameters (which can be changed in the same script) or with console arguments as follows:
```
python3 pi_HIW.py --hierarchical True --seed 1234 --env MontezumaRevengeNoFrameskip-v4 --atari-frameskip 15
```  
See the help (-h) section for more details.

For atari games, use the deterministic version of the gym environments, which can be specified by selecting v4 environments (e.g. "Breakout-v4"). Although the "NoFrameskip" environment is given, we set the frameskip anyway with parameter ```--atari-frameskip``` (15 in our experiments).

## Installation
* Install the [requirements](requirements.txt)
* Make sure that [gridenvs](https://github.com/aig-upf/gridenvs) and [pddl2gym](https://github.com/aig-upf/pddl2gym) are added to the python path.
