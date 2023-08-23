In this lesson, we've discussed an example of implementing a linear classifier with stochastic gradient descent.

Among all local optimization methods, this method is the easiest in implementation. Nevertheless, it is a powerful tool for tasks of this kind. We've considered a classification example with two classes – the presence or absence of type 2 diabetes. The algorithm may be used with multiple classes, too, although it will require considerable modifications.

In real-life tasks, you don't need to manually reproduce the algorithm from scratch – it has been implemented, for example, in the [scikit](https://scikit-learn.org/stable/modules/sgd.html) library. When using that version, you will need to pass the parameters we've discussed here to the function: the number of iterations for the "optimistic strategy" (`n_iter`) or the kind of the `loss` function. Among other things, the algorithm implemented in *scikit* has built-in settings to avoid exploding gradients.

The stochastic gradient descent method may be used for setting up the [Support Vector Machine method](https://en.wikipedia.org/wiki/Support-vector_machine) or even other linear classifiers and regressors. It is often used in text classification or in Natural Language Processing, as well as in setting up a single-layer perceptron or [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network). We will discuss it later in the course.