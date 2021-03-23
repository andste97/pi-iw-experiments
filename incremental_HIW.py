from IW import IW
import numpy as np
from tree_actor import EnvTreeActor, AbstractTreeActor
from planning_step import discover_atoms
import pandas as pd
import time
import random
from utils import immediate_subdirectories, natural_keys, InteractionsCounter


def restructure_tree(abstract_tree, new_abstract_feature):
    #First, we go through all abstract nodes and update their high level features
    for abstract_node in abstract_tree:
        if new_abstract_feature in abstract_node.low_level_tree.root.data["low_level_features"]:
            abstract_node.data["high_level_features"].add(new_abstract_feature)

    queue = abstract_tree._nodes.copy()
    while queue:
        abstract_node = queue.pop(0)

        AN_contains_AF = new_abstract_feature in abstract_node.data["high_level_features"]

        q = abstract_node.low_level_tree.root.children.copy()
        while q:
            n = q.pop(0)
            if n.entry_point is None:
                n_contains_AF = new_abstract_feature in n.data["low_level_features"]
                if (AN_contains_AF and not n_contains_AF) or (not AN_contains_AF and n_contains_AF):
                    new_tree = abstract_node.low_level_tree.detatch_subtree(n, copy_and_keep_node=True)
                    assert new_tree.root.depth == 0
                    assert n.entry_point is None
                    new_tree.root.entry_point = None
                    if AN_contains_AF:
                        af = abstract_node.data["high_level_features"]-set([new_abstract_feature])
                    else:
                        af = set([new_abstract_feature]).union(abstract_node.data["high_level_features"])
                    new_abstract_node = create_abstract_node(abstract_tree, abstract_node, af, n, new_tree)
                    queue.append(new_abstract_node)
                else:
                    q.extend(n.children)


def create_abstract_node(abstract_tree, abstract_parent, abstract_features, low_level_parent, low_level_tree):
    new_abstract_node = abstract_tree.add(abstract_parent,
                                          data={"done": low_level_tree.root.data["done"],
                                                "high_level_features": abstract_features})
    new_abstract_node.low_level_tree = low_level_tree
    new_abstract_node.low_level_parent = low_level_parent
    new_abstract_node.plan_initialized = False
    low_level_parent.entry_point = new_abstract_node

    abstract_children = abstract_parent.children.copy()
    for abstract_child in abstract_children:
        if abstract_child.low_level_parent in low_level_tree:
            abstract_parent.children.remove(abstract_child)
            new_abstract_node.children.append(abstract_child)
            abstract_child.parent = new_abstract_node
            abstract_child.update_depths(abstract_child.depth+1)

    return new_abstract_node


class FunctionCache:
    def __init__(self, fn):
        self.cached = dict()
        self.fn = fn

    def get_value(self, x):
        try:
            return self.cached[x]
        except KeyError:
            y = self.fn(x)
            self.cached[x] = y
            return y


def hierarchical_IW(seed, env, high_level_width, low_level_width, max_nodes):
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    applicable_actions_fn = lambda n: env.simulator.get_applicable_str_actions(n.data["s"])
    fc = FunctionCache(applicable_actions_fn)
    applicable_actions_fn = fc.get_value

    interactions = InteractionsCounter(budget=max_nodes)
    feature_extractor = FeatureExtractor()
    low_level_tree_actor = EnvTreeActor(env=env,
                                        applicable_actions_fn=applicable_actions_fn,
                                        observe_fns=[feature_extractor.get_features])

    # low_level_planner = RolloutIW(branching_factor=len(env_actions), width=low_level_width, features_name="low_level_features")
    low_level_planner = IW(generate_successor_fn=low_level_tree_actor.generate_successor, width=low_level_width, features_name="low_level_features", ignore_terminal_nodes=True)
    # low_level_planner = BFS(generate_successor_fn=low_level_tree_actor.generate_successor, features_name="low_level_features")

    class RewardObserver:
        def __init__(self):
            self.found = False
        def observe_node(self, node):
            if node.data["r"] is not None and node.data["r"] > 0:
                self.found = True

    reward_observer = RewardObserver()
    low_level_tree_actor.add_observe_fn(reward_observer.observe_node)
    low_level_tree_actor.add_observe_fn(lambda node: interactions.increment())

    low_level_planner.add_stop_fn(lambda tree: not interactions.within_budget())
    low_level_planner.add_stop_fn(lambda tree: reward_observer.found)

    abstract_tree_actor = AbstractTreeActor(low_level_planner=low_level_planner,
                                            low_level_tree_actor=low_level_tree_actor)

    high_level_planner = IW(generate_successor_fn=abstract_tree_actor.generate_successor, width=high_level_width, features_name="high_level_features", ignore_terminal_nodes=True)
    # high_level_planner = BFS(generate_successor_fn=abstract_tree_actor.generate_successor, features_name="high_level_features")

    high_level_planner.add_stop_fn(lambda tree: not interactions.within_budget())
    high_level_planner.add_stop_fn(lambda tree: reward_observer.found)

    return high_level_planner, abstract_tree_actor, reward_observer, feature_extractor


def get_unique_states(abstract_tree):
    return set([node.data["obs"] for abstract_node in abstract_tree if hasattr(abstract_node, "low_level_tree")
                for node in abstract_node.low_level_tree])


class FeatureExtractor:
    def __init__(self):
        self.abstract_atoms = set()
        self.nodes = list()

    def get_features(self, node):
        node.data["low_level_features"] = node.data["obs"]
        node.data["high_level_features"] = self.extract_abstract_features(node.data["low_level_features"]) # we add an empty string as a dummy atom
        self.nodes.append(node)

    def extract_abstract_features(self, features):
        s = self.abstract_atoms.intersection(features)
        s.add('')
        return s


if __name__ == "__main__":
    import os
    from pddl2gym.env import PDDLEnv
    from pddl2gym.simulator import PDDLProblemSimulator
    from pddl2gym.utils import parse_problem


    # HYPERPARAMETERS
    max_nodes = 10000
    seed = 0
    domains_path = "../pddl-benchmarks/"  # we assume domain is in the same directory as the instances (except for the case where different instances do not share the same domain)
    results_path = "results"  # one csv file per domain will be created in this directory
    domains = ["8puzzle"]  # add as many domains as you wish


    df = pd.DataFrame(columns=["domain_name", "ipc_domain", "domain_file","instance_file", "goal",
                        "w1_solved", "w1_nodes", "w1_states", "w1_time",
                        "w2_solved", "w2_nodes", "w2_states", "w2_time",
                        "H_solved", "H_nodes", "H_states", "H_abstract_nodes",
                        "H_time", "H_discover_time", "H_restructure_time"],
                      dtype=int)
    for domain in domains:
        domain_path = os.path.join(domains_path, domain)
        instances_path = domain_path
        if "domains" in immediate_subdirectories(domain_path):
            domain_path += "/domains"

        domain_files = [f for f in os.listdir(domain_path) if f.endswith(".pddl") and "domain" in f]
        domain_files.sort(key=natural_keys)
        instance_files = [f for f in os.listdir(instances_path) if f.endswith(".pddl") and not "domain" in f]
        instance_files.sort(key=natural_keys)

        df = pd.DataFrame(columns=["domain_path", "domain_file", "instance_file", "goal",
                                   "w1_solved", "w1_nodes", "w1_states", "w1_time",
                                   "w2_solved", "w2_nodes", "w2_states", "w2_time",
                                   "H_solved", "H_nodes", "H_states", "H_abstract_nodes",
                                   "H_time", "H_discover_time", "H_restructure_time"],
                          dtype=int)

        for i, instance_file in enumerate(instance_files):
            start_instance_time = time.time()
            if len(domain_files) == len(instance_files):
                domain_file = domain_files[i]
            else:
                assert "domain.pddl" in domain_files
                domain_file = "domain.pddl"

            print(f"\n{domain} {instance_file}", end=" ", flush=True)
            pddl_problem = parse_problem(os.path.join(domain_path, domain_file),
                                         os.path.join(instances_path, instance_file))
            complete_goal = pddl_problem.goal

            start_time = time.time()
            env = PDDLEnv(PDDLProblemSimulator(pddl_problem))
            ground_time = time.time() - start_time

            for goal_atom in complete_goal:
                goal_str = f"({goal_atom.name} {' '.join([p[0] for p in goal_atom.signature])})"
                env.simulator.change_goal(goal_atom)

                # IW(1)
                high_level_planner, abstract_tree_actor, reward_observer, fa = hierarchical_IW(seed=seed,
                                                                                               env=env,
                                                                                               high_level_width=1,
                                                                                               low_level_width=1,
                                                                                               max_nodes=max_nodes)
                start_w1_time = time.time()
                abstract_tree = abstract_tree_actor.reset()
                high_level_planner.initialize(tree=abstract_tree)
                high_level_planner.plan(tree=abstract_tree)

                w1_time = time.time() - start_w1_time
                w1_solved = reward_observer.found
                w1_nodes = abstract_tree_actor.get_tree_size(abstract_tree)

                print("O" if w1_solved else "X", end="", flush=True)
                assert len(abstract_tree) == 1

                w1_states = len(get_unique_states(abstract_tree))

                # IW(2)
                high_level_planner, abstract_tree_actor, reward_observer, _ = hierarchical_IW(seed=seed,
                                                                                              env=env,
                                                                                              high_level_width=1,
                                                                                              low_level_width=2,
                                                                                              max_nodes=max_nodes)
                start_w2_time = time.time()
                abstract_tree = abstract_tree_actor.reset()
                high_level_planner.initialize(tree=abstract_tree)
                high_level_planner.plan(tree=abstract_tree)

                w2_time = time.time() - start_w2_time
                w2_solved = reward_observer.found
                w2_nodes = abstract_tree_actor.get_tree_size(abstract_tree)

                print("O" if w2_solved else "X", end="", flush=True)
                assert len(abstract_tree) == 1
                w2_states = len(get_unique_states(abstract_tree))

                # HIW(1,1)
                high_level_planner, abstract_tree_actor, reward_observer, feature_extractor = \
                    hierarchical_IW(seed=seed, env=env, high_level_width=1, low_level_width=1, max_nodes=max_nodes)
                abstract_tree = abstract_tree_actor.reset()
                H_solved = False
                start_H_time = time.time()
                H_discover_time = 0
                H_restructure_time = 0
                discovered_atoms = set()
                candidate_atoms = list()
                while not H_solved:
                    high_level_planner.initialize(tree=abstract_tree)
                    high_level_planner.plan(tree=abstract_tree)

                    H_solved = reward_observer.found
                    H_nodes = abstract_tree_actor.get_tree_size(abstract_tree)

                    if not H_solved:
                        start_discover_time = time.time()
                        while len(candidate_atoms) == 0:
                            if len(feature_extractor.nodes) == 0:
                                break
                            candidate_atoms_used = feature_extractor.abstract_atoms
                            node = feature_extractor.nodes.pop()
                            discovered_atoms = discovered_atoms.union(discover_atoms(node, features_name="low_level_features"))
                            candidate_atoms = discovered_atoms - candidate_atoms_used
                        H_discover_time += time.time() - start_discover_time

                        if len(candidate_atoms) == 0:
                            break

                        start_restructure_time = time.time()
                        new_abstract_atom = candidate_atoms.pop()
                        feature_extractor.abstract_atoms.add(new_abstract_atom)
                        restructure_tree(abstract_tree, new_abstract_atom)
                        H_restructure_time += time.time() - start_restructure_time

                H_time = time.time() - start_H_time
                print("O" if H_solved else "X", end=" ", flush=True)

                H_states = len(get_unique_states(abstract_tree))

                total_time = time.time() - start_time

                df = df.append([{"domain_path": domain_path, "domain_file": domain_file, "instance_file": instance_file, "goal": goal_str,
                                 "w1_solved": w1_solved, "w1_nodes": w1_nodes, "w1_states": w1_states, "w1_time": w1_time,
                                 "w2_solved": w2_solved, "w2_nodes": w2_nodes, "w2_states": w2_states, "w2_time": w2_time,
                                 "H_solved": H_solved, "H_nodes": H_nodes, "H_states": H_states, "H_abstract_nodes": len(abstract_tree),
                                 "H_time": H_time, "H_discover_time": H_discover_time, "H_restructure_time": H_restructure_time,
                                 "ground_time": ground_time, "total_time": total_time}])
            os.makedirs(results_path, exist_ok=True)
            df.to_csv(os.path.join(results_path, f"{domain}.csv"))
