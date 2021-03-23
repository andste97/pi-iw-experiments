from gridenvs.examples.key_door import KeyDoorEnv
from gym import register

def mazeXL(**kwargs):
    init_map = ["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
                "W......W......W........W.......W",
                "W......W......W........W.......W",
                "W......W......W........W.......W",
                "W..............................W",
                "W......H...................D...W",
                "W.............W................W",
                "W.............W........W.......W",
                "WWW...WWW...WWWWWWWWWWWW......WW",
                "W.............W................W",
                "W.............W................W",
                "W.............W................W",
                "W.............W................W",
                "W.............W................W",
                "W.............W................W",
                "WWWWWWWWWW...WWWWWWWWWWW...WWWWW",
                "W.................W............W",
                "W.................W............W",
                "W..............................W",
                "W......W.......................W",
                "W......W.......................W",
                "W......W..........W............W",
                "W......W......WWWWWWW...WWWWWWWW",
                "W......W......W................W",
                "WWW...WWW...WWW................W",
                "W.............W................W",
                "W.............W........K.......W",
                "W..............................W",
                "W..............................W",
                "W.............W................W",
                "W.............W................W",
                "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"]
    return KeyDoorEnv(init_map, **kwargs)

def mazeL(**kwargs):
    init_map = ["WWWWWWWWWWWWWWWWWWWW",
                "W..................W",
                "W....H.........D...W",
                "W........W.........W",
                "W........W.........W",
                "WWW....WWWWW......WW",
                "W........W.........W",
                "W........W.........W",
                "W........W.........W",
                "W........W.........W",
                "W........W.........W",
                "W........W.........W",
                "WWWWW...WWWW...WWWWW",
                "W........W.........W",
                "W........W.........W",
                "W............K.....W",
                "W..................W",
                "W........W.........W",
                "W........W.........W",
                "WWWWWWWWWWWWWWWWWWWW"]
    return KeyDoorEnv(init_map, **kwargs)

for s in ["L", "XL"]:
    max_moves = 500 if s=="L" else 1000
    for r in [True, False]:
        register(id='GE_MKD%s-v%i'%(s, int(r)),
                 entry_point='envs:maze%s'%s,
                 kwargs={'max_moves': max_moves, "key_reward": r},
                 nondeterministic=False)

for s in ["L", "XL"]:
    for max_moves in [100, 200, 300, 400, 500, 1000]:
        register(id='GE_MKD%s%i-v0'%(s, max_moves),
                 entry_point='envs:maze%s'%s,
                 kwargs={'max_moves': max_moves, "key_reward": False},
                 nondeterministic=False)
