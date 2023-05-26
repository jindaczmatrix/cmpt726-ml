# -*- coding: utf-8 -*-
"""nonlinear_optimization_question.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zcNL0V4GR1bsPPqlfGR_9QYOnpxp_ZJI

# Nonlinear Optimization
This notebook provides a starter code for implementing nonlinear optimization methods for an objective function.

Please add this notebook to your Google Drive and complete all designated sections accordingly.

Creating a folder in your Google Drive to hold your Colaboratory assignments is recommended. Please include your .ipynb file with your assignment submission.

#Headers
Feel free to add any headers here.
"""

from math import sqrt
from numpy import asarray
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

"""# Objective Function
Modify the "objective" function using the following equation.

$$f(x,y) = 4x^{2} + \dfrac{(y+3)^2}{15}$$
"""

def objective(x, y):
	# <<TODO#1>> modify the objective function
	return 1

"""# Gradient of objective function
Modify the "gradient" function accordingly. 
"""

def gradient(x, y):
	# <<TODO#2>> modify the gradient function
	return asarray([1, 1])

"""#Part a: Plot the objective function
Use the following block to plot the objective function. Please do not modify it.
"""

# define range for input
bounds = asarray([[-15.0, 15.0], [-15.0, 15.0]])
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# Figure 1
# create a surface plot with the jet color scheme
figure_1 = pyplot.figure(1)
axis = figure_1.gca(projection='3d')
axis.plot_surface(x, y, objective(x, y), cmap='jet')

"""#Part b: Basic Gradient Descent

## b.1. Set Parameters
You may change the following parameters.
"""

# define the total iterations
n_iter = 200
# steps size
gamma_t = 0.2

"""##b.2. Gradient Descent Function
Modify the following function to implement a basic gradient descent approach.
"""

def gradient_descent(objective, derivative, bounds, n_iter, gamma_t):
	# track all solutions
	solutions = list()
	# Consider an initial point
	p = [-10, 10]
	score = objective(p[0], p[1])
	# run the gradient descent
	for t in range(n_iter):
		# report progress
		print('>%d f(%s) = %.5f' % (t, p, score))
		#<<TODO#3>> Add your code here
	return solutions

"""##b.3. Solve the Problem using Gradient Descent
Use the following block to solve the problem. Please do not modify it.
"""

# perform the gradient descent search
solutions = gradient_descent(objective, gradient, bounds, n_iter, gamma_t)
solutions = asarray(solutions)

"""##b.4. Plot results for Basic Gradient Descent
Use the following block to plot the results. Please do not modify it.
"""

# Figure 2
# create a filled contour plot with 50 levels and jet color scheme
figure_2 = pyplot.figure(2)
pyplot.contourf(x, y, objective(x, y), levels=50, cmap='jet')
# plot the solution
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()

"""#Part c: Adaptive Gradient (AdaGrad)

##c.1. Set Parameters
You may change the following parameters.
"""

# define the total iterations
n_iter = 200
# define the step size
gamma_t = 0.2

"""##c.2. AdaGrad Function
Modify the following function to implement the AdaGrad approach.
"""

def adagrad(objective, derivative, bounds, n_iter, gamma_t):
	# track all solutions
	solutions = list()
	# Consider an initial point
	p = [-10, 10]
	score = objective(p[0], p[1])
	# initialize the preconditioner
	D = [0.0 for _ in range(bounds.shape[0])]
	# run the AdaGrad
	for t in range(n_iter):
		# report progress
		print('>%d f(%s) = %.5f' % (t, p, score))
		#<<TODO#4>> Add your code here
	return solutions

"""##c.3. Solve the Problem using AdaGrad
Use the following block to solve the problem. Please do not modify it. 
"""

# perform the AdaGrad search
solutions = adagrad(objective, gradient, bounds, n_iter, gamma_t)
solutions = asarray(solutions)

"""## c.4. Plot results for AdaGrad
Use the following block to plot the results. Please do not modify it.
"""

# Figure 3
# create a filled contour plot with 50 levels and jet color scheme
figure_3 = pyplot.figure(3)
pyplot.contourf(x, y, objective(x, y), levels=50, cmap='jet')
# plot the sample as black circles
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()

"""#Part d: Adam

##d.1. Set Parameters
You may change the following parameters.
"""

eps=1e-8
# define the total iterations
n_iter = 200
# steps size
gamma_t = 0.2
# factor for average gradient
alpha = 0.8
# factor for average squared gradient
beta = 0.999

"""##d.2. Adam Function
Modify the following function to implement the Adam approach.
"""

def adam(objective, derivative, bounds, n_iter, gamma_t, alpha, beta, eps):
	solutions = list()
	# consider an initial point
	p = [-10, 10]
	score = objective(p[0], p[1])
	# initialize first and second moments
	m = [0.0 for _ in range(bounds.shape[0])]
	D = [0.0 for _ in range(bounds.shape[0])]
	# run the Adam updates
	for t in range(n_iter):
		#<<TODO#5>> Add your code here
		# report progress
		print('>%d f(%s) = %.5f' % (t, p, score))
	return solutions

"""##d.3. Solve the Problem using Adam
Use the following block to solve the problem. Please do not modify it. 
"""

# perform the Adam search
solutions = adam(objective, gradient, bounds, n_iter, gamma_t, alpha, beta, eps)
solutions = asarray(solutions)

"""## d.4. Plot results for Adam
Use the following block to plot the results. Please do not modify it.
"""

# Figure 4
# create a filled contour plot with 50 levels and jet color scheme
figure_4 = pyplot.figure(4)
pyplot.contourf(x, y, objective(x, y), levels=50, cmap='jet')
# plot the solution
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()