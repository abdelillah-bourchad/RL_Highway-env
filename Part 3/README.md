## Training an Agent
Training is done using `training_A2C.py`, `training_DQN.py` and `training_PPO.py`.
You can set the parameters directly on the files. Executing the scripts will create two folders logs and models.

## Visualizing Agent Behavior

To visualize the behavior of an agent, follow these steps:

1. Open the `plot_model.py` script.
2. Set the `model_path` variable to the path of your saved model.
3. Use the appropriate function (`A2C`, `PPO`, or `DQN`) to load the model.

Example:
```python
model_path = "path/to/your/model"
model = A2C.load(model_path)
```

## Using Tensorboard

In order to visualize the training plots, use Tensorboard:

1. Open a Terminal.
2. Run the following command, replacing `<logs_directory>` with the path to your logs directory:

```bash
tensorboard --logdir=<logs_directory>
```
