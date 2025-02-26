import numpy as np

class Network:
    def __init__ (self, architecture, activation_function, loss_function):
        self.architecture = architecture
        self.weights = []
        self.biases = []
        self.activations = []
        self.activation_function = activation_function
        self.loss_function = loss_function

    def initialize_weights(self, weights):
        self.weights = weights

    #build weight and bias layers
    def build_layers(self):
        for i in range(len(self.architecture)-1):
            weight_layer = np.random.randn(self.architecture[i], self.architecture[i+1])
            self.weights.append(weight_layer)
            bias_layer = np.random.randn(1, self.architecture[i+1])
            self.biases.append(bias_layer)
    
    #Activation functions
    def sigmoid(self, z):
        return 1/(1+ np.exp(-z))
    
    #loss_functions
    def MSE(self, A_o, labels, m):
        return (1/m) * np.sum(np.sum((np.abs(A_o - labels))**2, axis=1))

    #function to pass data through network
    def nn_pass(self, data):
        act_g = data
        self.activations.append(act_g)
        for i in range(len(self.weights)):
            if self.activation_function == "sigmoid":
                z = act_g@self.weights[i]+self.biases[i]
                act_g = self.sigmoid(z)
                self.activations.append(act_g)
        
        return act_g
    
    #produce loss for the network
    def nn_loss(self, labels, m):
        if self.loss_function == "MSE":
            return self.MSE(self.activations[len(self.activations)-1], labels, m)
        
    def grad(self, m, labels):
        propagators = []
        dA_dz_list = []
        dz_dA_list = self.weights
        dL_dw_list = []
        dL_db_list = []

        final_activation = self.activations[len(self.activations)-1]

        if self.activation_function == "sigmoid" and self.loss_function == "MSE":
            dL_dAfinal = (2/m)*(final_activation - labels)

        for i in self.activations:
            dA_dz = i*(1-i)
            dA_dz_list.append(dA_dz)
            
        dL_dz = dL_dAfinal * dA_dz_list[len(dA_dz_list)-1]
        propagators.append(dL_dz)
        #calculator propagators
        prop = 0
        for i in range(len(dz_dA_list)-1, 0, -1):
            prop = dL_dz @ dz_dA_list[i].T
            dL_dz = prop * dA_dz_list[i]
            propagators.append(dL_dz)

        #claculate weight gradients
        for i in range(0, len(propagators)):
            dL_dw = self.activations[i].T @ list(reversed(propagators))[i]
            dL_dw_list.append(dL_dw)
            
        #calculate bias_gradients
        for i in range(len(propagators)):
            dL_db = np.sum(propagators[i], axis=0)
            dL_db_list.append(dL_db)

        #calculate dL_dSoftmax

        return dL_dw_list, dL_db_list

    def optimize_weights_biases(self, w_grad, b_grad, learn_rate, over_shoot=False, loss = 1):
        for i in range(len(self.weights)-1):
            if over_shoot == True:
                self.weights[i] -= w_grad[i]*learn_rate
                self.biases[i] -= list(reversed(b_grad))[i]*learn_rate
            else:
                self.weights[i] -= w_grad[i]*learn_rate*loss
                self.biases[i] -= list(reversed(b_grad))[i]*learn_rate*loss
            

    def backpropagate(self, m, labels, learn_rate, overshoot=False, loss=1):
        gradient = self.grad(m, labels)
        self.optimize_weights_biases(gradient[0], gradient[1], learn_rate, overshoot, loss)
        self.activations = []

