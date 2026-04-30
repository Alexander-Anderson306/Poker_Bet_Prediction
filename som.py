import numpy as np
import matplotlib.pyplot as plt
from sklearn_som.som import SOM
from data_prep import load_class_data
from data_prep import prepare_persona_predictor_card_info
from data_prep import balance_and_limit_samples
from joblib import Parallel, delayed
import itertools

#seed
np.random.seed(67)

def train_and_score_som_config(X_train, size, epochs, lr):
    som = train_som(X_train, size, epochs=epochs, lr=lr)
    qe = quantization_error(som, X_train)
    return {
        "som": som,
        "size": size,
        "epochs": epochs,
        "lr": lr,
        "qe": qe
    }



#quantization error calculation
#the lower the distance between the data points and their BMUs, the better the SOM is at representing the data
def quantization_error(som, X):
    #bmus best matching units for each input point
    bmus = som.predict(X)
    weights = som.weights.reshape(-1, som.dim)

    #calculates the average euclidean distance between each point and its BMU
    total_distance = 0

    for i in range(len(X)):
        #the data point
        x = X[i]
        #which node represents it
        bmu_index = bmus[i]  
        #that node's weight vector        
        w = weights[bmu_index]       

        #distance between them
        distance = np.linalg.norm(x - w)  

        total_distance += distance

    return total_distance / len(X)


#train a SOM of a given size
def train_som(X_train, size, epochs=10, lr=0.5):
    som = SOM(
        m=size,
        n=size,
        dim=X_train.shape[1],
        lr=lr,
        sigma=1.0,
        random_state=67
    )

    som.fit(X_train, epochs=epochs, shuffle=True)
    return som


#visualize the hit map of the SOM
def save_som_hit_map(som, X, size, save_plot_name):
    labels = som.predict(X)

    hit_map = np.zeros((size, size))

    for label in labels:
        row = label // size
        col = label % size
        hit_map[row, col] += 1

    plt.figure()
    plt.imshow(hit_map)
    plt.colorbar(label="Number of samples")
    plt.title("Best SOM Hit Map")
    plt.xlabel("SOM column")
    plt.ylabel("SOM row")

    plt.savefig(f"PHOTOS/{save_plot_name}", dpi=300, bbox_inches="tight")
    plt.close()


#Raw feature SOM
def optimize_raw_som(X_train):
    #hyper parameters
    map_sizes = [6, 8, 10, 12, 15]
    epoch_values = [5, 10, 20, 40]
    lr_values = [0.05, 0.1, 0.2, 0.4, 0.6]

    configs = list(itertools.product(map_sizes, epoch_values, lr_values))

    print(f"Running {len(configs)} SOM configurations in parallel...")

    results = Parallel(n_jobs=4, backend="threading", verbose=10)(
        delayed(train_and_score_som_config)(X_train, size, epochs, lr)
        for size, epochs, lr in configs
    )

    #find best
    best_result = min(results, key=lambda r: r["qe"])

    best_som = best_result["som"]
    size = best_result["size"]
    epochs = best_result["epochs"]
    lr = best_result["lr"]
    best_qe = best_result["qe"]

    print("\nBest configuration:")
    print(f"Size: {size}x{size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Quantization error: {best_qe:.5f}")

    #save SOM visualization
    save_som_hit_map(best_som, X_train, size, "best_raw_som_hit_map.png")

    return best_som, size, epochs, lr, best_qe

#load full dataset
df = load_class_data('CSVs/poker_data_clustered_FULL.csv')

# balance classes / limit samples
df = balance_and_limit_samples(df)

# prepare features + split + scaling
X_train, X_test, y_train, y_test = prepare_persona_predictor_card_info(df)

print(f"Training samples: {X_train.shape}")
print(f"Testing samples:  {X_test.shape}")

# run SOM optimization
best_som, size, epochs, lr, best_qe = optimize_raw_som(X_train)

print("\nFinal result:")
print(f"Best SOM size: {size}x{size}")
print(f"Epochs: {epochs}")
print(f"Learning rate: {lr}")
print(f"Quantization error: {best_qe:.5f}")