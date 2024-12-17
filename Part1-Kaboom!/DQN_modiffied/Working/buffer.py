import numpy as np  # For numerical operations

class SumTree:
    def __init__(self, capacity):
        """
        Initialize the SumTree with a given capacity.

        Args:
          capacity: The maximum number of experiences to store in the tree.
        """
        self.capacity = capacity  # Number of leaf nodes (max number of experiences)
        self.tree = np.zeros(2 * capacity - 1)  # Total number of nodes in the tree
        self.data = np.zeros(capacity, dtype=object)  # Stores the actual experiences
        self.write = 0  # Pointer to where the next experience will be written
        self.n_entries = 0  # Total number of experiences stored

    def _propagate(self, idx, change):
        """
        Update the tree by propagating the change in priority up to the root.

        Args:
          idx: The index of the leaf node whose priority has changed.
          change: The amount by which the priority has changed.
        """
        parent = (idx - 1) // 2  # Calculate the parent node index
        self.tree[parent] += change  # Update the parent's priority

        if parent != 0:  # If not at the root
            self._propagate(parent, change)  # Recursively propagate the change

    def update(self, idx, priority):
        """
        Update the priority of a given leaf node and propagate the change.

        Args:
          idx: The index of the leaf node to update.
          priority: The new priority of the leaf node.
        """
        change = priority - self.tree[idx]  # Calculate the change in priority
        self.tree[idx] = priority  # Update the leaf node's priority
        self._propagate(idx, change)  # Propagate the change up the tree

    def add(self, priority, data):
        """
        Add a new experience with its priority to the tree.

        Args:
          priority: The priority of the new experience.
          data: The experience to add.
        """
        idx = self.write + self.capacity - 1  # Calculate the leaf index
        self.data[self.write] = data  # Store the experience in the data array
        self.update(idx, priority)  # Update the tree with the new priority

        self.write += 1  # Move the write pointer to the next position
        if self.write >= self.capacity:
            self.write = 0  # Overwrite if capacity is exceeded

        if self.n_entries < self.capacity:  # Increment the count of entries
            self.n_entries += 1

    def _retrieve(self, idx, s):
        """
        Find the leaf node that corresponds to the given cumulative sum s.

        This method traverses the tree from the root to a leaf, 
        guided by the cumulative sums stored in the tree nodes.

        Args:
          idx: The index of the current node being examined.
          s: The cumulative sum to search for.

        Returns:
          The index of the leaf node corresponding to the cumulative sum s.
        """
        left = 2 * idx + 1  # Index of the left child
        right = left + 1  # Index of the right child

        if left >= len(self.tree):  # If reached a leaf node
            return idx

        if s <= self.tree[left]:  # If s is less than or equal to the left child's sum
            return self._retrieve(left, s)  # Search in the left subtree
        else:  # Otherwise
            return self._retrieve(right, s - self.tree[left])  # Search in the right subtree

    def get(self, s):
        """
        Get the experience corresponding to the cumulative sum s.

        Args:
          s: The cumulative sum to search for.

        Returns:
          A tuple containing the leaf index, priority, and the experience data.
        """
        idx = self._retrieve(0, s)  # Find the leaf index
        dataIdx = idx - self.capacity + 1  # Calculate the index in the data array
        return (idx, self.tree[idx], self.data[dataIdx])  # Return the experience

    @property
    def total(self):
        """
        Get the total priority (sum of all priorities).

        Returns:
          The total priority stored in the root of the tree.
        """
        return self.tree[0]  # The root node holds the total priority


class ExperienceReplay:
    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        """
        Initialize the Experience Replay buffer with Prioritized Experience Replay (PER).

        Args:
          capacity: The maximum number of experiences to store.
          alpha: The priority exponent (controls how much prioritization is used).
          epsilon: A small constant added to the TD error to avoid zero priorities.
        """
        self.tree = SumTree(capacity)  # Use a SumTree to store experiences with priorities
        self.alpha = alpha  # Exponent for prioritizing experiences
        self.epsilon = epsilon  # Small constant to prevent zero priorities

    def append(self, experience, td_error=1.0):
        """
        Add a new experience to the replay buffer.

        Args:
          experience: The experience tuple (state, action, reward, done, next_state).
          td_error: The initial TD error of the experience.
        """
        priority = (abs(td_error) + self.epsilon) ** self.alpha  # Calculate priority
        self.tree.add(priority, experience)  # Add the experience to the SumTree

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences from the replay buffer with PER.

        Args:
          batch_size: The number of experiences to sample.
          beta: The importance sampling exponent (controls how much bias correction is used).

        Returns:
          A tuple containing the batch of experiences and importance sampling weights.
        """
        
        # This line seems to have no effect, it's just a comparison
        # self.tree.total == 0 

        batch = []
        idxs = []
        segment = self.tree.total / batch_size  # Divide the total priority into segments
        priorities = []

        for i in range(batch_size):
            a = segment * i  # Start of the segment
            b = segment * (i + 1)  # End of the segment
            s = np.random.uniform(a, b)  # Sample a value within the segment
            idx, priority, data = self.tree.get(s)  # Get the experience based on the sampled value
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Calculate Importance Sampling (IS) weights
        # Add a small constant to avoid division by zero
        sampling_probabilities = priorities / (self.tree.total + 1e-8)  
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta) 
        is_weight /= is_weight.max()  # Normalize weights
        is_weight = np.array(is_weight, dtype=np.float32)  # Convert to float32

        # Unpack experiences for easier use
        states, actions, rewards, dones, next_states, _ = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
            is_weight,  # Importance sampling weights
            idxs  # Indices of the sampled experiences in the tree
        )

    def __len__(self):
        """
        Get the number of experiences stored in the replay buffer.

        Returns:
          The number of experiences in the buffer.
        """
        return self.tree.n_entries

    def update_priorities(self, idxs, td_errors):
        """
        Update the priorities of the experiences with the given indices.

        Args:
          idxs: A list of indices of experiences in the SumTree.
          td_errors: A list of corresponding TD errors for each experience.
        """
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha  # Recalculate priority
            self.tree.update(idx, priority)  # Update the priority in the SumTree