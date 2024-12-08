import matplotlib.pyplot as plt

def plot_results(model, X_test, y_test):
    predictions = model.predict(X_test)
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Irrigation')
    plt.ylabel('Predicted Irrigation')
    plt.title('Actual vs Predicted Irrigation')
    plt.savefig('plots/plot.png')  # Save the plot to the 'plots' directory
    plt.show()
