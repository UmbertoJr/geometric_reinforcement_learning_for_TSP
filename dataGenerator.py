import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


class DataGenerator(object):

    def __init__(self):
        pass
    
    def create_upper_matrix(self, values, size):
        upper = np.zeros((size, size))
        r = np.arange(size)
        mask = r[:,None] < r
        upper[mask] = values
        
        return upper 

    
    def gen_instance(self, max_length, dimension, seed=0): # Generate random TSP instance
        sequence = np.random.rand(max_length, dimension) # (max_length) cities with (dimension) coordinates in [0,1]
        distance = self.create_upper_matrix( pdist(sequence, "euclidean") , max_length)
        distance = distance.T + distance
        
        return sequence, distance
        

    def train_batch(self, batch_size, max_length, dimension): # Generate random batch for training procedure
        input_batch = []; positions_batch = []        
        
        for _ in range(batch_size):
            pos_ , dist_ = self.gen_instance(max_length, dimension) # Generate random TSP instance
            input_batch.append(dist_); positions_batch.append(pos_)
            
        return np.array(positions_batch), np.array(input_batch)

    
    def test_batch(self, batch_size, max_length, dimension, seed=0): # Generate random batch for testing procedure
        
        distances_batch = []; positions_batch = []
        
        if seed!=0: np.random.seed(seed)
        
        for _ in range(batch_size): 
            pos_, dist_ = self.gen_instance(max_length, dimension) # Generate random TSP instance
            distances_batch.append(dist_ ); positions_batch.append(pos_)
            
        return np.array(positions_batch), np.array(distances_batch)

    
    def reward(self, tsp_sequence, distance_matrix):
        tot_dist_path = 0
        for j in range(tsp_sequence.shape[0]):
            tot_dist_path += distance_matrix[tsp_sequence[j-1], tsp_sequence[j]]
        return tot_dist_path

    # Swap city[i] with city[j] in sequence
    def swap2opt(self, tsp_sequence,i,j):
        new_tsp_sequence = np.copy(tsp_sequence)
        new_tsp_sequence[i:j+1] = np.flip(tsp_sequence[i:j+1], axis=0) # flip or swap ?
        return new_tsp_sequence

    # One step of 2opt = one double loop and return first improved sequence
    def step2opt(self, tsp_sequence,  matrix_dist):
        seq_length = tsp_sequence.shape[0]
        best_tsp_sequence  = np.copy(tsp_sequence)
        distance = self.reward(tsp_sequence, matrix_dist)
        for i in range(1,seq_length-1):
            for j in range(i+1,seq_length):
                new_tsp_sequence = self.swap2opt(tsp_sequence,i,j)
                new_distance = self.reward(new_tsp_sequence, matrix_dist)
                if new_distance < distance:
                    best_tsp_sequence  = np.copy(new_tsp_sequence)
                    distance = new_distance
        return best_tsp_sequence, distance


    def loop2opt(self, tsp_sequence, matrix_dist,  max_iter=400): # Iterate step2opt max_iter times (2-opt local search)
        #print("##### start to loop2opt")
        
        best_reward = self.reward(tsp_sequence, matrix_dist)
        new_tsp_sequence = np.copy(tsp_sequence)
        vist = 0
        for it in range(max_iter): 
            #if it%50 == 0:
                #print(it, best_reward)
            new_tsp_sequence, new_reward = self.step2opt(new_tsp_sequence, matrix_dist)
            if new_reward < best_reward:
                best_reward = new_reward
            elif new_reward == best_reward:
                vist += 1
                if vist ==5:
                    #print("the best reward is : ", best_reward)
                    return new_tsp_sequence, best_reward
        
        #print("the best reward is : ", best_reward)
        return new_tsp_sequence, best_reward

    def visualize_2D_trip(self, trip): # Plot tour
        plt.figure(1)
        colors = ['red'] # First city red
        for _ in range(len(trip)-1):
            colors.append('blue')
            
        plt.scatter(trip[:,0], trip[:,1],  color=colors) # Plot cities
        tour=np.array(list(range(len(trip))) + [0]) # Plot tour
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    