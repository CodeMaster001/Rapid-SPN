import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset("iris")

g = sns.PairGrid(iris, y_vars=["sepal_length"], x_vars=["sepal_width","sepal_width"], height=2)
g.map(plt.scatter, color=".3",edgecolor="white")
g.set(ylim=(-1, 11), yticks=[0, 5, 10]);
plt.show()
