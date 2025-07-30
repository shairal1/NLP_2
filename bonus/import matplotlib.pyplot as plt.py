# manually generate the plot for the report because of the long running time (2 days). The values are from the current output. 
import matplotlib.pyplot as plt

epochs = list(range(1, 11))

loss_nl2en = [6.1002, 5.4590, 5.1835, 4.9865, 4.8377, 4.7201, 4.6134, 4.5005, 4.4295, 4.3554]
loss_en2sv = [6.4194, 5.7329, 5.4232, 5.2034, 5.0221, 4.8732, 4.7376, 4.6381, 4.5287, 4.4379]

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_nl2en, label='NL → EN', marker='o')
plt.plot(epochs, loss_en2sv, label='EN → SV', marker='s')
plt.title('Training Loss per Epoch (Pivot Models)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pivot_training_loss.png')
plt.show()

