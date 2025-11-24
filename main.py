import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


X = []
y = []

with open(r"C:\Users\user\Documents\Course_Folder\ML\data.txt", "r") as file: #the file was attached
    for line in file:
        line = line.strip()
        parts = line.split(",")
        X.append(float(parts[0]))        # sqft living area
        y.append(float(parts[2]))        # price


X = np.array(X)
y = np.array(y)
y = y / 1000000  #normalize the Y value


plt.scatter(X, y, label="Data")
plt.legend()
plt.show()

w = 0.0
b = 0.0
lr = 0.00000005
epochs = 2000

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range (m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    return cost / (2 * m)


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    return dj_dw / m, dj_db / m


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w, b = w_in, b_in
    J_history = []
    p_history = []

    for i in range(num_iters + 1):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        J_history.append(compute_cost(x, y, w, b))
        p_history.append([w, b])

        if i % max(1, num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.9f}, w: {w:0.9f}, b: {b:0.9f}")

    return w, b, J_history, p_history



w_final, b_final, J_hist, p_hist = gradient_descent(
    X, y, w, b, lr, epochs)

final_cost = compute_cost(X, y, w_final, b_final)

print("\n--- Calculating result ---")
print(f"Optimal (w): {w_final:0.9f}")
print(f"Optimal (b): {b_final:0.9f}")
print(f"Minimum Cost (J): {final_cost:0.7f}")

y_pred = w_final * X + b_final

plt.figure(figsize=(10, 6))

plt.scatter(X, y, color="blue", marker="o", label="Data")

plt.plot(X, y_pred, color="red", label=f"Regresion line (w={w_final:.2f}, b={b_final:.2f})")

plt.title("Result of linear regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()
