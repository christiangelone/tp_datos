import operator as op
import math as m
import random as r

class Network(object):

    #public
    def __init__(self,dim_input,dim_output,learn_factor=1.0,name='Perceptron Network'):
        self.id = r.randint(1,999999)
        self.name = name
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.prediction = None

        dim_hidden = int(m.sqrt(self.dim_input * self.dim_output))
        self.hlayer = Layer(dim_hidden,self.dim_input,learn_factor,'(Hidden) Layer',1)

        self.olayer = OutputLayer(self.dim_output,dim_hidden,learn_factor,'(Output) Layer',2)

    def description(self):
        desc = "\n  " + self.name + str(self.id) + " loaded "
        desc += "(Input dimensions = " + str(self.dim_input) + ", "
        desc += "Output dimensions = " + str(self.dim_output) + "):\n\n"
        desc += self.hlayer.description()
        desc += self.olayer.description()
        desc += "Prediction -> " + str(self.prediction)
        print desc
        return self

    #Hace el forward propagation del input
    def work(self,input,verbose=False):
        if verbose: print "Network working..."
        h_output = self.hlayer.work(input,verbose)
        self.prediction = self.olayer.work(h_output,verbose)
        if verbose: print "Done!"
        return self

    def predict(self):
        return self.prediction

    def update(self,desire_output,verbose=False):
        if verbose: print "Updating network..."

        delta_prop = self.olayer.update(desire_output,verbose)
        self.hlayer.update(delta_prop,verbose)

        if verbose: print "Done!"
        return self

class Layer(object):

    #public
    def __init__(self,q_neurons,dim_input,learn_factor,name = 'Layer',id=r.randint(1,999999)):
        self.id = id
        self.name = name
        self.learn_factor = learn_factor
        self.neurons = []
        self.input = None
        self.output = None
        self.output_prime = None

        for n in range(0,q_neurons):
            neuron = Neuron(dim_input,n)
            self.neurons.append(neuron)

    def description(self):
        desc =  "        " + self.name  + " " + str(self.id) + " loaded("
        desc += "Neurons = " +  str(len(self.neurons)) + ","
        desc += "Learn factor = " +  str(self.learn_factor) + "):\n\n"
        for neuron in self.neurons:
            desc += neuron.description()
        return desc

    def work(self,input,verbose=False):
        if verbose: print "    Layer " + str(self.id) + ": Neurons working..."
        self.input = input
        self.output = []
        self.output_prime = []

        for neuron in self.neurons:
            output, output_prime = neuron.work(self.input,verbose)
            self.output.append(output)
            self.output_prime.append(output_prime)

        if verbose: print "    Done!"
        return self.output

    def update(self,delta_prop,verbose=False):
        if verbose: print "    Layer " + str(self.id) + ": Updating neurons..."

        if verbose: print "     Calculating delta..."
        delta = map(op.mul,delta_prop,self.output_prime) # hadamard product

        if verbose: print "     Calculating delta_back..."
        delta_back = []
        for i in range(0,len(self.input)):
            weights_j = map(lambda x: x.weight(i),self.neurons) #weight_i of all neurons
            delta_back_k = sum(map(op.mul,weights_j,delta)) #inner product with delta
            delta_back.append(delta_back_k)

        i = 0
        for neuron in self.neurons:
            delta_weight = map(lambda x: - (self.learn_factor * delta[i] * x), self.input)
            neuron.update(delta_weight,verbose)
            i+=1

        if verbose: print "    Done!"
        return delta_back

class OutputLayer(Layer):

    def update(self,desire_output,verbose=False):
        if verbose: print "    Layer " + str(self.id) + ": Updating neurons..."

        if verbose: print "     Calculating delta..."
        delta_output = map(op.sub,self.output,desire_output) # output difference
        delta = map(op.mul,delta_output,self.output_prime) # hadamard product

        if verbose: print "     Calculating delta_back..."
        delta_back = []
        for i in range(0,len(self.input)):
            weights_j = map(lambda x: x.weight(i),self.neurons) #weight_i of all neurons
            delta_back_k = sum(map(op.mul,weights_j,delta)) #inner product with delta
            delta_back.append(delta_back_k)

        if verbose: print "     Applying delta_weight..."
        i = 0
        for neuron in self.neurons:
            delta_weight = map(lambda x: - (self.learn_factor * delta[i] * x), self.input)
            neuron.update(delta_weight,verbose)
            i+=1

        if verbose: print "    Done!"
        return delta_back

class Neuron(object):

    #public
    def __init__(self,dim_input,id=r.randint(1,999999)):
        self.id = id
        self.input = None
        self.weights = []
        self.output = None
        self.output_prime = None

        for n in range(0,dim_input):
            weight = r.uniform(0.0, 0.25)
            self.weights.append(weight)

    def description(self):
        desc =  "           Neuron " + str(self.id) + " loaded:\n"
        # desc += "               Input   -> " +  str(self.input) + "\n"
        desc += "               Weights -> " +  str(self.weights) + "\n"
        desc += "               Output  -> " +  str(self.output) + "\n\n"
        return desc

    def weight(self,j):
        return self.weights[j]

    def work(self,input,verbose=False):
        if verbose: print "        Neuron " + str(self.id) + " working..."
        self.input = input
        z_output = sum(map(op.mul,self.input,self.weights)) #inner product

        self.output = self.__activation(z_output)
        self.output_prime = self.__activation_prime(z_output)

        if verbose: print "         Input -> " + str(self.input)
        if verbose: print "         Output -> " + str(self.output)
        if verbose: print "        Done!"
        return (self.output,self.output_prime)

    def update(self,delta_weights,verbose=False):
        if verbose: print "         Updating neuron " + str(self.id) + "..."

        if verbose: print "          Updating weights... "
        self.weights = map(sum, zip(self.weights,delta_weights)) #Adding error

        if verbose: print "         Done!"
        return self

    #private
    def __activation(self,value):
        sigmoid = None
        try:
            sigmoid = 1.0 / (1.0 + m.exp(value * -1.0))
        except OverflowError:
            sigmoid = 0

        return sigmoid

    def __activation_prime(self,value):
        sigmoid_prime = None
        try:
            sigmoid_prime = self.__activation(value) * (1.0 - self.__activation(value))
        except OverflowError:
            sigmoid_prime = 0

        return sigmoid_prime

# ----------------------------  debug  ----------------------------------------
# network = Network(4,4,learn_factor=1)
# network.work([1,1,1,1]).description().update([1,0,0,0]).description()
# print network.predict()
