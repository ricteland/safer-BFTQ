# safer-BFTQ

This repository provides a re-implementation of the Budgeted Fitted Q-iteration (BFTQ) algorithm, a model-free reinforcement learning algorithm for safe exploration. The project is designed to be a testbed for different uncertainty quantification methods in deep reinforcement learning, including Bayesian Neural Networks (BNN), Monte Carlo Dropout, and Deep Ensembles.

## Getting Started

### Dependencies

This project uses [Nix](https://nixos.org/) to manage dependencies. The `flake.nix` file defines the complete development environment. The main dependencies are:

- Python 3.12
- PyTorch
- Pyro
- Gymnasium
- highway-env
- NumPy
- SciPy

### Installation

1.  **Install Nix:** If you don't have Nix installed, follow the instructions [here](https://nixos.org/download.html).
2.  **Enable Flakes:** Make sure you have flakes enabled in your Nix configuration.
3.  **Start the development shell:** Open a terminal in the project root and run:

    ```bash
    nix develop
    ```

    This will drop you into a shell with all the necessary dependencies installed.

## Usage

### Training

To run the BFTQ agent, use the `train_baseline.py` script:

```bash
python train_baseline.py
```

To run the BNN-BFTQ agent, use the `train_bnn.py` script:

```bash
python train_bnn.py
```

This will start the training process with the default configuration.

### Logging

This project uses the standard Python `logging` module and TensorBoard for logging.

- **Console and File Logs:** The logs are printed to the console and saved to a file in the `logs/` directory.
- **TensorBoard:** TensorBoard logs are saved in the `logs/tensorboard/` and `logs/tensorboard_bnn/` directories. To view them, run:

  ```bash
  tensorboard --logdir logs/
  ```

## Project Structure

- `agents/`: Contains the implementation of the BFTQ agent.
  - `bftq/`: The core logic of the BFTQ algorithm.
- `models/`: Contains the neural network models.
  - `q_net.py`: A simple budgeted Q-network.
  - `bnn.py`: A Bayesian Q-network using Pyro.
  - `ensemble.py`: (Not yet implemented) A deep ensemble of Q-networks.
  - `mc_dropout.py`: (Not yet implemented) A Q-network with Monte Carlo Dropout.
- `utils/`: Contains utility functions.
  - `logger.py`: The logging configuration.
  - `replay_buffer.py`: The replay buffer implementation.
- `train_baseline.py`: The main script to run the training.
- `train_bnn.py`: The main script to run the BNN-BFTQ training.
- `flake.nix`: The Nix flake defining the development environment.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
