import network
import mnist_loader

print("Loading MNSIT data...")
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
source_data = list(mnist_loader.read("training"))
training_data = source_data[:50000]
validation_data = source_data[50001:60000]
test_data = list(mnist_loader.read("testing"))
print("Loaded {0} samples for training and {1} samples for testing.".format(len(training_data), len(test_data)))

print("Creating network...")
net = network.Network([784, 30, 10], cost = network.CrossEntropyCost)
print("Initalizing large weights")
#net.large_weight_initializer()
print("Using stochastic grad desc to train...")
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)
