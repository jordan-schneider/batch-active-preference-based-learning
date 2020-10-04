This code elicits preferences between trajectories from humans, distills those preferences into a
minimal alignment test, and then tests a number of randomly generated agents.

To run:
1. Run the following once if the ctrl_samples directory is empty or does not currently exist.
```python
	python input_sampler.py driver 500000
``` 
1. Run the following to start the inference procedure. You should press both `a` and `b` every iteration to watch the trajectories, and press either `1` or `2` to select the trajectory you like more. Press `space` to start each video, and press `Esc` to exit the video and return to the prompt.
```python
	python demos.py --N=110 --M=1000 --b=10 --outdir=human/[subject-id]
```

If at any time you get bored/want to give up, press `Ctrl-C` EXACTLY ONCE and wait a moment to save
your progress. I'd really appreciate if you could give 110 preferences, but any amount is fine.

I haven't implemented hot-reloading. If you've accidentally exited midway and want to resume, let me
know and I can throw something together. It would be best if you could just keep the terminal open
and/or do it all in one sitting.


This code is a fork of https://github.com/Stanford-ILIAD/batch-active-preference-based-learning

The novel contributions are in post.py and run_test.py. 


---

This code learns reward functions from human preferences in various tasks by actively generating batches of scenarios and querying a human expert.

Companion code to [CoRL 2018 paper](https://arxiv.org/abs/1810.04303):  
E Bıyık, D Sadigh. **"[Batch Active Preference-Based Learning of Reward Functions](https://arxiv.org/abs/1810.04303)"**. *Conference on Robot Learning (CoRL)*, Zurich, Switzerland, Oct. 2018.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [MuJoCo 1.50](http://www.mujoco.org/index.html)
- [NumPy](https://www.numpy.org/)
- [OpenAI Gym](https://gym.openai.com)
- [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home)
- PYMC
- [Scikit-learn](https://scikit-learn.org)
- [SciPy](https://www.scipy.org/)
- [theano](http://deeplearning.net/software/theano/)

## Running
Throughout this demo,
- [task_name] should be selected as one of the following: Driver, LunarLander, MountainCar, Swimmer, Tosser
- [method] should be selected as one of the following: nonbatch, greedy, medoids, boundary_medoids, successive_elimination, random
For the details and positive integer parameters K, N, M, b, B; we refer to the publication.
You should run the codes in the following order:

### Sampling the input space
This is the preprocessing step, so you need to run it only once (subsequent runs will overwrite for each task). It is not interactive and necessary only if you will use batch active preference-based learning. For non-batch version and random querying, you can skip this step.

You simply run
```python
	python input_sampler.py [task_name] K
```
For quick (but highly suboptimal) results, we recommend K=1000. In the article, we used K=500000.

### Learning preference reward function
This is where the actual algorithms work. You can simply run
```python
	python run.py [task_name] [method] N M b
```
b is required only for batch active learning methods. We fixed B=20b. To change that simply go to demos.py and modify 11th line.
Note: N must be divisible by b.
After each query or batch, the user will be showed the w-vector learned up to that point. To understand what those values correspond to, one can check the 'Tasks' section of the publication.

### Demonstration of learned parameters
This is just for demonstration purposes. run_optimizer.py starts with 3 parameter values. You can simply modify them to see optimized behavior for different tasks and different w-vectors.
