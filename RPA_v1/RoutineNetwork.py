import numpy as np
from toolkit import *
import time
from collections import defaultdict


class RoutineNetwork:

    def __init__(self, N):

        self.N = N

        self.history_matrix = np.zeros((self.N, self.N))
        self.markov_matrix = np.zeros((self.N, self.N ))
        self.adjacency_matrix = np.zeros((self.N, self.N)).astype(np.int8)
        self.slice_memory = defaultdict(int)
        self.automated_routine_sequence = {}
        self.automated_routine_memory = set([])

    def update_matrix(self, new_sequence, last_sequence):

        # add new routine
        for cur in range(len(new_sequence)-1):
            self.history_matrix[new_sequence[cur]][new_sequence[cur+1]] += 1

        # forget last routine
        if last_sequence is not None:
            for cur in range(len(last_sequence)-1):
                self.history_matrix[last_sequence[cur]][last_sequence[cur+1]] -= 1

        self.markov_matrix = np.zeros(self.history_matrix.shape)

        for row in range(len(self.markov_matrix)):
            if np.sum(self.history_matrix[row]) == 0:
                continue
            self.markov_matrix[row] = self.history_matrix[row]/np.sum(self.history_matrix[row])

        self.adjacency_matrix = np.zeros(self.markov_matrix.shape).astype(np.int8)
        self.adjacency_matrix[np.where(self.markov_matrix > 0)] = 1

    def query_next_node(self, v, current_event):

        if np.random.uniform(0, 1) < v:
            next_event = np.random.choice(np.arange(current_event+1, self.N))

        else:

            transition_count_vector = self.history_matrix[current_event]
            possible_event_count = [transition_count_vector[x] for x in np.arange(current_event+1, self.N)]

            if np.sum(possible_event_count)>0:
                p = [x/np.sum(possible_event_count) for x in possible_event_count]
                next_event = np.random.choice(np.arange(current_event+1, self.N), p=p)
            else:
                next_event = np.random.choice(np.arange(current_event+1, self.N))

        return next_event

    def generate_new_sequence(self, v):

        sequence = [0]

        loop = False

        while sequence[-1] != self.N -1:

            sequence.append(
                self.query_next_node(v, sequence[-1])
            )

            if len(sequence)>5*self.N:
                loop = True
                break

        if not loop:
            return sequence
        else:
            return self.generate_new_sequence(v)

    def measure_possible_path(self, ):
        count = 0
        stack = [(0, [0])]
        path_union = []

        while len(stack) > 0:
            node, path = stack.pop()

            for cur in range(self.N):
                if self.adjacency_matrix[node][cur] == 1 and cur not in path:
                    new_path = list(path)
                    if cur != self.N-1:
                        new_path.append(cur)
                        stack.append((cur, new_path))
                    else:
                        new_path.append(cur)
                        count += 1
                        path_union.append(list(new_path))
        return count, path_union

    def measure_change_magnitude(self, adjacent_matrix_1, adjacent_matrix_2):

        return np.sum(np.bitwise_xor(adjacent_matrix_1, adjacent_matrix_2))

    def measure_cluster_coefficient(self, ):
        res = []

        for cur in range(self.N+1):

            neighbor = []

            for nei in range(self.N+1):
                if self.adjacency_matrix[cur][nei] == 1:
                    neighbor.append(nei)

            if len(neighbor) == 0 or len(neighbor) == 1:
                res.append(0)
            else:
                denominator = len(neighbor)*(len(neighbor)-1)//2

                nominator = 0
                for nei in neighbor:
                    for inner_cur in range(self.N+1):
                        if self.adjacency_matrix[nei][inner_cur] == 1 and inner_cur in neighbor:
                            nominator += 1
                res.append(nominator/denominator)
        return np.mean(res)

    def update_matrix_according_to_rpa(self, ):

        for source in self.automated_routine_sequence:

            # print(self.automated_routine_sequence[source])
            for cur in range(len(self.automated_routine_sequence[source])-1):
                self.markov_matrix[self.automated_routine_sequence[source][cur]] = np.zeros(self.N)
                self.markov_matrix[
                    self.automated_routine_sequence[source][cur]
                ][
                    self.automated_routine_sequence[source][cur+1]
                ] = 1
        self.adjacency_matrix = np.zeros(self.markov_matrix.shape).astype(np.int8)
        self.adjacency_matrix[np.where(self.markov_matrix > 0)] = 1


def simulation(return_dic, idx, repeat, period, N, V, R, F, L):

    ress_change_magnitude = []
    ress_complexity = []
    # ress_structure = []

    for r in range(repeat):

        if r % 1 == 0:
            print(r, time.asctime())

        # res_change_magnitude = []
        res_complexity = []
        # res_structure = []

        network = RoutineNetwork(N)

        memory_stack = []

        initial_sequence = np.arange(0, N).tolist()
        memory_stack.append(list(initial_sequence))
        network.update_matrix(initial_sequence, None)

        slices = analyze_routine(initial_sequence, L)

        for slice in slices:

            network.slice_memory[
                "".join(slice)
            ] += 1

        for step in range(period):

            # if step>0 and step%100==0:
            #     print(step, V, R, F, L)

            next_sequence = network.generate_new_sequence(V,)
            if len(memory_stack) < R:
                memory_stack.append(list(next_sequence))
                last_sequence = None
            else:
                memory_stack.append(list(next_sequence))
                last_sequence = memory_stack.pop(0)
            network.update_matrix(next_sequence, last_sequence)

            slices = analyze_routine(next_sequence, L)
            for slice in slices:
                network.slice_memory[
                    "".join(slice)
                ] += 1

            if last_sequence is not None:
                slices = analyze_routine(last_sequence, L)
                for slice in slices:
                    network.slice_memory[
                        "".join(slice)
                    ] -= 1

            for key in network.slice_memory:
                if network.slice_memory[key] / R >=F:

                    if key in network.automated_routine_memory:
                        continue
                    else:
                        sequence = decode_sequence(key)
                        network.automated_routine_sequence[sequence[0]] = sequence
                        network.automated_routine_memory.add(key)
                        break

            network.update_matrix_according_to_rpa()

            count, path_union = network.measure_possible_path()
            # print(count)
            res_complexity.append(count)

        # ress_change_magnitude.append(res_change_magnitude)
        # ress_structure.append(res_structure)
        ress_complexity.append(res_complexity)

    return_dic[idx] = (
        ress_complexity
    )

