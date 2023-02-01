from __future__ import print_function

import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt
import random
from itertools import product

plt.style.use("seaborn-darkgrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
sns.set(rc={'axes.facecolor':'darkgrey', 'figure.facecolor':'darkgrey'})

tf.config.list_physical_devices('GPU') 



def larn(params):
	random.seed(42)

	learning_rate, num_iterations = params
	# the optimizer allows us to apply gradients to update variables
	optimizer = tf.keras.optimizers.Adam(learning_rate)

	# Create a fixed matrix, A
	# A = tf.random.normal([4,4])
	A = tf.convert_to_tensor(tf.constant([
		[3,4,7,3],
		[3,6,8,9],
		[7,5,8,2],
		[6,3,8,5]
	], tf.float32), name="a num")

	# B = tf.random.normal([4,4])

	B = tf.convert_to_tensor(tf.constant([
		[1,4,1,2],
		[3,2,5,6],
		[5,1,1,2],
		[2,5,3,4]
	], tf.float32), name="b num")

	# Create x using an arbitrary initial value
	# x = tf.convert_to_tensor(tf.constant([[2],[3],[2],[4]], tf.float32), name="aa")
	x = tf.Variable(tf.ones([4,1]))
	y = tf.Variable(tf.ones([4,1]))
	# Create a fixed vector b
	# b = tf.random.normal([4,1])
	# c = tf.random.normal([4,1])

	c = tf.convert_to_tensor(tf.constant([2,3,2,4], tf.float32), name="banana")

	# Check the initial values
	# print("A:", A.numpy())
	# print("B:", B.numpy())
	# print("c:", c.numpy())

	# print("Initial x:", x.numpy())
	# print("Ax:", (A @ x).numpy())
	# print("Bx:", (B @ y).numpy())
	# print()
	# print("Ax^2", ((A @ x)**2).numpy())
	# print("Bx^2", ((B @ y)**2).numpy())
	# print()
	# print("sqrt( (xA)^2 + (xB)^2 )", tf.sqrt( (A @ x)**2 + (B @ y)**2 ).numpy())

	error = []

	# We want sqrt( (xA)^2 + (yB)^2 ) - C = 0, so we'll try to minimize its value
	for step in range(num_iterations):
		# print("Iteration", step)
		with tf.GradientTape() as tape:
			# Calculate A*x
			product = tf.sqrt( (A @ x)**2 + (B @ y)**2 )
			# calculat the loss value we want to minimize
			# what happens if we don't use the square here?
			difference_sq = tf.math.square(product - c)
			error.append(tf.norm(tf.math.sqrt(difference_sq)).numpy())
			# print("Squared error:", error[-1])
			# calculate the gradient
			grad = tape.gradient(difference_sq, [x,y])
			# print("Gradients:")
			# print(grad)
			# update x
			optimizer.apply_gradients(zip(grad, [x,y]))
			# print()
	return error

# Check the final values
# print("Optimized x", x.numpy())
# print("Optimized y", y.numpy())
# print("c", c.numpy())
# print("sqrt( (xA)^2 + (xB)^2 )", tf.sqrt( (A @ x)**2 + (B @ y)**2 ).numpy())  # Should be close to the value of b
l_rate = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
num_i = [10, 50, 100, 500, 1000]
form = list(product(l_rate, num_i))

plt.figure(figsize=(30,40), dpi=100)
e = [[],[],[],[],[]]
sns.set(rc={"figure.figsize":(8, 4)})
for index, item in enumerate(form):
    e[index%5].append(larn(item))
# sns.lineplot(e)

# sns.lineplot(error)
# plt.legend([],[], frameon=False)
fig, axs = plt.subplots(3,2)
# for ax in axs:
# axs.get_legend().remove()
for i, a in enumerate(e):
	sns.lineplot(a, ax=axs[i//2,i%2], legend=None)