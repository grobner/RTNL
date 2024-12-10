# Randomized tensor networks reservoir computing: validity and learnability phase transitions

Shinji Sato, Daiki Sasaki, Chih-Chieh Chen, Kodai Shiba, Tomah Sogabe

## Environment

The scripts use [pytorch](https://pytorch.org), [Google/TensorNetwork](https://github.com/google/TensorNetwork) and [cupy](https://cupy.dev).
This script uses GPU. Check your CUDA version and install the corresponding version of cupy and pytorch.
The libraries can be installed by
```
pip install jupyter numpy matplotlib pandas tensornetwork tqdm umap-learn scikit-learn
```

## Information

In our experiment, we used [Kubota's code](https://github.com/kubota0130/ipc) to calculate IPC. All the codes in [ipc_calculate/utils/](ipc_calculate/utils/) are exactly the same as theirs.

## Experiment
The experiment is divided into five folders.

- [learn static hand-written data](learn_static_data/README.md)
- [Information Processing Capacity](ipc_calculate/README.md)
- [Lorenz equation](chaos_prediction/README.md)
- [NARMA5](NARMA/README.md)
- [Z500](weather_report/README.md)
- [sunspot](sunspot/README.md)
