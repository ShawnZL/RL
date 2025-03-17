class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def select(self, c_puct):
        pass

    def expand(self, action_priors):
        pass

    def update(self, leaf_value):
        pass

    def update_recursive(self, leaf_value):
        pass

    def get_value(self, c_puct):
        pass

    def is_leaf(self):
        pass

    def is_root(self):
        pass


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self.root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self.c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move
            action, node = node.select(self.c_puct)
            state.do_move(action)
            # Evaluate the leaf using a network which outputs a list of (action, probability)
            # tuples p and also a score v in [-1, 1] for the current player.
            action_probs, leaf_value = self._policy(state)

            # Check for end of game.
            end, winner = state.game.end()
            if not end:
                node.expand(action_probs)
            else:
                if winner == -1:
                    leaf_value = 0.0
                else:
                    leaf_value = 1.0 if winner == state.get_current_player() else -1.0

            node.update_recursive(-leaf_value)
