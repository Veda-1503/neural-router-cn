import numpy as np
import networkx as nx
import random


TOPOLOGIES = ['barabasi_albert', 'erdos_renyi', 'watts_strogatz']


class NetworkEnvironment:
    def __init__(self, num_nodes=15, malicious_ratio=0.2, seed=None,
                 randomize_graph=True, forced_topology=None):
        self.num_nodes = num_nodes
        self.num_malicious = int(num_nodes * malicious_ratio)
        self.seed = seed
        self.randomize_graph = randomize_graph
        # forced_topology: one of TOPOLOGIES or None (→ random each rebuild)
        self.forced_topology = forced_topology if forced_topology in TOPOLOGIES else None
        self.graph = None
        self.malicious_nodes = set()
        self.source = None
        self.destination = None
        self.current_node = None
        self.visited = set()
        self.max_steps = num_nodes * 3
        self.steps = 0
        self._valid_pairs_cache = None
        self._build_graph()

    def _build_graph(self, episode_seed=None):
        s = episode_seed if episode_seed is not None else (self.seed if self.seed else random.randint(0, 99999))
        rng = random.Random(s)

        # Use forced topology or pick randomly
        topology = self.forced_topology if self.forced_topology else rng.choice(TOPOLOGIES)

        attempts = 0
        while True:
            attempts += 1
            curr_seed = s + attempts
            try:
                if self.num_nodes < 5:
                    self.graph = nx.complete_graph(self.num_nodes)
                elif topology == 'barabasi_albert':
                    m = int(rng.randint(1, max(1, min(self.num_nodes - 1, 3))))
                    self.graph = nx.barabasi_albert_graph(self.num_nodes, m, seed=curr_seed)
                elif topology == 'erdos_renyi':
                    p = rng.uniform(0.35, 0.65)
                    self.graph = nx.erdos_renyi_graph(self.num_nodes, p, seed=curr_seed)
                else:  # watts_strogatz
                    k = min(self.num_nodes - 1, rng.randint(4, 6))
                    p = rng.uniform(0.15, 0.40)
                    self.graph = nx.watts_strogatz_graph(self.num_nodes, int(k), p, seed=curr_seed)

                if self.graph and nx.is_connected(self.graph):
                    break
            except Exception:
                topology = 'barabasi_albert'

        pos_dict = nx.spring_layout(self.graph, seed=s)
        nx.set_node_attributes(self.graph, pos_dict, 'pos')

        all_nodes = list(self.graph.nodes())
        self.malicious_nodes = set(rng.sample(all_nodes, self.num_malicious))

        for node in self.graph.nodes():
            if node in self.malicious_nodes:
                self.graph.nodes[node]['trust'] = round(rng.uniform(0.1, 0.6), 2)
            else:
                self.graph.nodes[node]['trust'] = round(rng.uniform(0.4, 0.9), 2)

        # Invalidate pair cache whenever graph is rebuilt
        self._valid_pairs_cache = None

    def _get_valid_pairs(self):
        if self._valid_pairs_cache is not None:
            return self._valid_pairs_cache

        safe_nodes = [n for n in self.graph.nodes() if n not in self.malicious_nodes]
        subgraph = self.graph.subgraph(safe_nodes)
        pairs = []
        for src in safe_nodes:
            for dst in safe_nodes:
                if src == dst:
                    continue
                try:
                    if len(nx.shortest_path(subgraph, src, dst)) >= 2:
                        pairs.append((src, dst))
                except nx.NetworkXNoPath:
                    continue

        self._valid_pairs_cache = pairs
        return pairs

    def get_valid_actions(self):
        neighbors = list(self.graph.neighbors(self.current_node))
        unvisited = [n for n in neighbors if n not in self.visited]
        return unvisited  # Return empty list if stuck — act() handles None gracefully

    def get_action_features(self, action_node):
        trust    = self.graph.nodes[action_node]['trust']
        is_dest  = 1.0 if action_node == self.destination else 0.0
        degree   = len(list(self.graph.neighbors(action_node))) / self.num_nodes
        pos_a    = np.array(self.graph.nodes[action_node]['pos'])
        pos_d    = np.array(self.graph.nodes[self.destination]['pos'])
        dist_norm = float(np.linalg.norm(pos_a - pos_d))
        steps_norm = self.steps / self.max_steps
        return np.array([trust, is_dest, degree, dist_norm, steps_norm], dtype=np.float32)

    def get_state(self):
        valid = self.get_valid_actions()
        return valid, [self.get_action_features(a) for a in valid]

    def reset(self, episode=None, is_test=False):
        if self.randomize_graph:
            rebuild = is_test or (episode is not None and episode % 10 == 0) or self.graph is None
            if rebuild:
                ep_seed = episode if episode is not None else random.randint(0, 99999)
                self._build_graph(episode_seed=ep_seed)

        valid_pairs = self._get_valid_pairs()
        if not valid_pairs:
            self._build_graph()
            valid_pairs = self._get_valid_pairs()

        self.source, self.destination = random.choice(valid_pairs)
        self.current_node = self.source
        self.visited = {self.source}
        self.steps = 0
        return self.get_state()

    def step(self, action_node):
        self.steps += 1
        self.current_node = action_node
        self.visited.add(action_node)

        if action_node in self.malicious_nodes:
            return self.get_state(), -50, True, {"reason": "malicious"}
        elif action_node == self.destination:
            speed_bonus = max(0, 5 - self.steps)
            return self.get_state(), 10 + speed_bonus, True, {"reason": "success"}
        elif self.steps >= self.max_steps:
            return self.get_state(), -10, True, {"reason": "timeout"}
        else:
            trust_bonus = 0.1 * self.graph.nodes[action_node]['trust']
            return self.get_state(), -1 + trust_bonus, False, {"reason": "step"}