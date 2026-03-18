import matplotlib.pyplot as plt

def plot_cost(cost_history):
    plt.plot(cost_history)
    plt.title('Rzeczywiste vs Przewidywane')
    plt.xlabel('Przewidywane')
    plt.ylabel('Rzeczywiste')
    plt.show()
