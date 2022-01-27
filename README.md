# pub

Miscellaneous public code. 

## qlearn.py 

Simple discrete, deterministic Q learning. 

Can be run in command line, or in notebook (e.g. colab).

The implemented environment example is from Tom Mitchell's machine learning textbook (1997). See http://www.cs.cmu.edu/~tom/mlbook.html .

* Environment: All actions in the goal state G loop back, with reward 0. <br/> <img src="/images/grid.png" width="200px">
* Optimal V(s):  <br/> <img src="/images/value.png" width="200px">
* Optimal Q(s,a):  <br/> <img src="/images/finalQ.png" width="200px">

Note: V(s) and Q(s,a) assume a discount rate of 0.9.

## spiral.ipynb

Simple script to generate the spiral data. 

Change two variables under "config" to adjust the data. 

