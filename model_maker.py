import numpy as np
import glob
import cv2

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


#prepare data
#glioma_tumors
glioma_tumor = []
path1 = 'C:\\Users\\svene\\OneDrive\\Documents\Desktop\\My Programming Projects\\Projects\\Natural Language Model\\glioma_tumor\\*.jpg'
for f in glob.iglob(path1):
    image = cv2.imread(f)
    image = cv2.resize(image, (256, 256)).flatten()
    glioma_tumor.append(image)

#meningioma
meningioma_tumor = []
path2 = 'C:\\Users\\svene\\OneDrive\\Documents\\Desktop\\My Programming Projects\\Projects\\Natural Language Model\\meningioma_tumor\\*.jpg'
for f in glob.iglob(path2):
    image = cv2.imread(f)
    image = cv2.resize(image, (256, 256)).flatten()
    meningioma_tumor.append(image)
    print(np.shape(image))

#no
no_tumor = []
path3 = 'C:\\Users\\svene\\OneDrive\\Documents\\Desktop\\My Programming Projects\\Projects\\Natural Language Model\\no_tumor\\*.jpg'
for f in glob.iglob(path3):
    image = cv2.imread(f)
    image = cv2.resize(image, (256, 256)).flatten()
    no_tumor.append(image)

#pituitary
pituitary_tumor = []
path4 = 'C:\\Users\\svene\\OneDrive\\Documents\\Desktop\\My Programming Projects\Projects\\Natural Language Model\\pituitary_tumor\\*.jpg'
for f in glob.iglob(path4):
    image = cv2.imread(f)
    image = cv2.resize(image, (256, 256)).flatten()
    pituitary_tumor.append(image)

#prepare labels
print(len(glioma_tumor)//7)

#glioma
glioma_labels = []
for i in range(len(glioma_tumor)):
    glioma_labels.append(np.array([1, 0, 0, 0]))

#meningioma
meningioma_labels = []
for i in range(len(meningioma_tumor)):
    meningioma_labels.append(np.array([0, 1, 0, 0]))

#no
no_labels = []
for i in range(len(no_tumor)):
    no_labels.append(np.array([0, 0, 1, 0]))

#pituitary
pituitary_labels = []
for i in range(len(pituitary_tumor)):
    pituitary_labels.append(np.array([0, 0, 0, 1]))

#build network
my_nn = Network([196608, 300, 300, 300, 4], "sigmoid", "MSE")
my_nn.build_layers()
print(my_nn.weights)

#pass data in batches of 7
n = 0
a = 0


print(my_nn.weights)
for i in range(len(glioma_tumor)):
        my_nn.nn_pass(np.array([glioma_tumor[i], meningioma_tumor[i]]))
        l = my_nn.nn_loss([[1, 0, 0, 0], [0, 1, 0, 0]], 2)
        print(l)
        print(my_nn.activations[len(my_nn.activations)-1])
        n+=7
        my_nn.backpropagate(2, [[1, 0, 0, 0], [0, 1, 0, 0]], 1, l)
            
        print("p")
    
#my_nn.nn_pass(glioma_tumor)


   