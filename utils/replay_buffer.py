import torch
import numpy as np
import random
import psutil


class ReplayBuffer:
    def __init__(self, max_buffer_size, num_last_frames=4, frame_shape=[1, 84, 84], crash_if_no_mem=True):
        self.max_buffer_size = max_buffer_size
        self.num_last_frames_to_fetch = num_last_frames
        self.frame_height = frame_shape[1]
        self.frame_width = frame_shape[2]
        self.current_free_slot_index = 0
        self.current_buffer_size = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.frames = np.zeros(frame_shape + [max_buffer_size], dtype=np.uint8)
        self.actions = np.zeros((max_buffer_size, 1), dtype=np.uint8)
        self.rewards = np.zeros((max_buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_buffer_size, 1), dtype=np.uint8)

        self._check_enough_ram(crash_if_no_mem)

    def store_frame(self, frame):
        self.frames[self.current_free_slot_index] = frame
        self.current_free_slot_index = (self.current_free_slot_index + 1) % self.max_buffer_size
        self.current_buffer_size = min(self.max_buffer_size, self.current_buffer_size + 1)

        return self.current_free_slot_index - 1

    def store_effect(self, index, action, reward, done):
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

    def fetch_random_states(self, batch_size):
        assert self._has_enough_data(batch_size), f"Not enough data to fetch, expected {batch_size}"

        random_indices = random.sample(range(self.max_buffer_size - 1), batch_size)

        states = self._preprocess_states(
            np.concatenate([self._fetch_state(i) for i in random_indices], 0)
        )

        next_states = self._preprocess_states(
            np.concatenate([self._fetch_state(i + 1) for i in random_indices], 0)
        )

        actions = torch.from_numpy(self.actions[random_indices]).to(self.device).long()
        rewards = torch.from_numpy(self.rewards[random_indices]).to(self.device)
        dones = torch.from_numpy(self.dones[random_indices]).to(self.device).float()

        return states, actions, rewards, next_states, dones

    def _has_enough_data(self, batch_size):
        return batch_size < self.current_buffer_size

    def _preprocess_states(self, state):
        return torch.from_numpy(state).to(self.device).float().div(255)

    def _fetch_state(self, end_index):
        end_index += 1
        start_index = end_index - self.num_last_frames_to_fetch
        start_index = self._handle_start_index_edge_cases(start_index, end_index)
        num_missing_frames = self.num_last_frames_to_fetch - (end_index - start_index)

        if start_index < 0 or num_missing_frames > 0:
            state = [np.zeros_like(self.frames[0]) for _ in range(num_missing_frames)]

            for index in range(start_index, end_index):
                state.append(self.frames[index % self.max_buffer_size])

            # shape = (C, H, W) -> (1, C, H, W) where C - number of past frames, 4 for Atari
            return np.concatenate(state, 0)[np.newaxis, :]
        else:
            # reshape from (C, 1, H, W) to (1, C, H, W) where C number of past frames, 4 for Atari
            return self.frames[start_index:end_index].reshape(-1, self.frame_height, self.frame_width)[np.newaxis, :]

    def _check_enough_ram(self, crash_if_no_mem):
        def to_GBs(memory_in_bytes):
            return memory_in_bytes / 2 ** 30

        available_memory = psutil.virtual_memory().available
        required_memory = self.frames.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        print(f'required memory = {to_GBs(required_memory)} GB, available memory = {to_GBs(available_memory)} GB')

        if required_memory > available_memory:
            message = f"Not enough memory to store the complete replay buffer! \n" \
                      f"required: {to_GBs(required_memory)} > available: {to_GBs(available_memory)} \n" \
                      f"Page swapping will make your training super slow once you hit your RAM limit." \
                      f"You can either modify replay_buffer_size argument or set crash_if_no_mem to False to ignore it."
            if crash_if_no_mem:
                raise Exception(message)
            else:
                print(message)

    def _handle_start_index_edge_cases(self, start_index, end_index):
        # Edge case 1:
        # Index is "too close" to 0 and our circular buffer is still not full, thus we don't have enough frames
        if not self._buffer_full() and start_index < 0:
            start_index = 0

        # Edge case 2:
        # Handle the case where start index crosses the buffer head pointer - the data before and after the head pointer
        # belongs to completely different episodes
        if self._buffer_full():
            if 0 < (self.current_free_slot_index - start_index) % self.max_buffer_size < self.num_last_frames_to_fetch:
                start_index = self.current_free_slot_index

        # Edge case 3:
        # A done flag marks a boundary between different episodes or lives either way we shouldn't take frames
        # before or at the done flag into consideration
        for index in range(start_index, end_index - 1):
            if self.dones[index % self.max_buffer_size]:
                start_index = index + 1

        return start_index

    def _buffer_full(self):
        return self.current_buffer_size == self.max_buffer_size


if __name__ == '__main__':
    replay_buffer = ReplayBuffer(50000)
    print(replay_buffer.fetch_random_states(16))
