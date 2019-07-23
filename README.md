# Reinforcement Learning Algorithm

## Value based
- [Q Learning](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/Q_learning) 

<a href="https://www.codecogs.com/eqnedit.php?latex=q^{T}(s,a)=q^{T-1}(s,a)&space;&plus;\frac&space;1&space;N[r(s')&plus;&space;\max_{a'}q^{T-1}(s',a')-q^{T-1}(s,a)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q^{T}(s,a)=q^{T-1}(s,a)&space;&plus;\frac&space;1&space;N[r(s')&plus;&space;\max_{a'}q^{T-1}(s',a')-q^{T-1}(s,a)]" title="q^{T}(s,a)=q^{T-1}(s,a) +\frac 1 N[r(s')+ \max_{a'}q^{T-1}(s',a')-q^{T-1}(s,a)]" /></a>

- [Sarsa](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/sarsa) 

<a href="https://www.codecogs.com/eqnedit.php?latex=q^{T}(s,a)=q^{T-1}(s,a)&space;&plus;\frac&space;1&space;N[r(s')&plus;&space;q^{T-1}(s',a')-q^{T-1}(s,a)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q^{T}(s,a)=q^{T-1}(s,a)&space;&plus;\frac&space;1&space;N[r(s')&plus;&space;q^{T-1}(s',a')-q^{T-1}(s,a)]" title="q^{T}(s,a)=q^{T-1}(s,a) +\frac 1 N[r(s')+ q^{T-1}(s',a')-q^{T-1}(s,a)]" /></a>

- [Sarsa Lambda](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/sarsa_lambda) 

<a href="https://www.codecogs.com/eqnedit.php?latex=q^{T}(s,a)=q^{T-1}(s,a)&space;&plus;\frac&space;1&space;N[r(s')&plus;&space;q^{T-1}(s',a')-q^{T-1}(s,a)]E(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q^{T}(s,a)=q^{T-1}(s,a)&space;&plus;\frac&space;1&space;N[r(s')&plus;&space;q^{T-1}(s',a')-q^{T-1}(s,a)]E(s,a)" title="q^{T}(s,a)=q^{T-1}(s,a) +\frac 1 N[r(s')+ q^{T-1}(s',a')-q^{T-1}(s,a)]E(s,a)" /></a> 

E(s,a) is eligibility trace.

- [DQN](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/DQN/DQN-tf)