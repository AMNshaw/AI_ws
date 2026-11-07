from micrograd.nn import MLP

n = MLP(3, [4, 4, 1])
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]


for k in range(20):
    y_pred = [n(x) for x in xs]
    loss = sum((y_pred[i] - ys[i])**2 for i in range(len(ys)))

    n.zero_grad()

    loss.backward()
    for p in n.parameters():
        p.data -= 0.05 * p.grad

    print(f"step [{k}] current loss: {loss.data}")