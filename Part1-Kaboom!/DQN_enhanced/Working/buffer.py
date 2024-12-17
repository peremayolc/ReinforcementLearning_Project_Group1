import numpy as np  # For numerical operations

class SumTree:
    def __init__(self, capacity):
        """
        Initialize the SumTree with a given capacity.
        """
        self.capacity = capacity  # Number of leaf nodes (max number of experiences)
        self.tree = np.zeros(2 * capacity - 1)  # Total number of nodes in the tree
        self.data = np.zeros(capacity, dtype=object)  # Stores the actual experiences
        self.write = 0  # Pointer to where the next experience will be written
        self.n_entries = 0  # Total number of experiences stored

    def _propagate(self, idx, change):
        """
        Update the tree by propagating the change up to the root.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """
        Update the priority of a given leaf node and propagate the change.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        """
        Add a new experience with its priority to the tree.
        """
        idx = self.write + self.capacity - 1  # Calculate the leaf index
        self.data[self.write] = data  # Store the experience
        self.update(idx, priority)  # Update the tree with the new priority

        self.write += 1  # Move the write pointer
        if self.write >= self.capacity:
            self.write = 0  # Overwrite if capacity is exceeded

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _retrieve(self, idx, s):
        """
        Find the leaf node that corresponds to the cumulative sum s.
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        """
        Get the experience corresponding to the cumulative sum s.
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    @property
    def total(self):
        """
        Get the total priority.
        """
        return self.tree[0]


class ExperienceReplay:
    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon

    def append(self, experience, td_error=1.0):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)


    def sample(self, batch_size, beta=0.4):
        self.tree.total == 0

        batch = []
        idxs = []
        segment = self.tree.total / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Add a small constant to avoid division by zero
        sampling_probabilities = priorities / (self.tree.total + 1e-8)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        is_weight = np.array(is_weight, dtype=np.float32)

        # Unpack experiences
        states, actions, rewards, dones, next_states, _ = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
            is_weight,
            idxs
        )

    def __len__(self):
        return self.tree.n_entries

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)