import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        #self.states = []
        self.depth_map = []
        self.angle = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        #n_states = len(self.states)
        n_states = len(self.depth_map)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return  np.array(self.depth_map),\
                np.array(self.angle),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, depth_map, angle, action, probs, vals, reward, done):
        #self.states.append(state)
        self.depth_map.append(depth_map)
        self.angle.append(angle)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        #self.states = []
        self.depth_map = []
        self.angle = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
