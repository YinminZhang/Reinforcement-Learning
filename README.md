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

![](http://latex.codecogs.com/gif.latex?Y_t%5E%7B%5Ctext%7BDQN%7D%7D%20%5Cequiv%20R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20Q%28S_%7Bt&plus;1%7D%2C%20a%3B%20%5Ctheta_%7Bt_%7B%5Ctext%7Bold%7D%7D%7D%29)
- [Double DQN](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/Double_DQN/Double_DQN-tf)

![](http://latex.codecogs.com/gif.latex?Y_t%5E%7B%5Ctext%7BDoubleDQN%7D%7D%20%5Cequiv%20R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20Q%28S_%7Bt&plus;1%7D%2C%20%5Cmathop%20%7Bargmax%7D_%7Ba%7DQ%28S_%7Bt&plus;1%7D%2Ca%3B%5Ctheta_t%29%2C%20%5Ctheta_%7Bt_%7B%5Ctext%7Bold%7D%7D%7D%29)

- [Dueling DQN](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/Dueling_DQN)

![](http://latex.codecogs.com/gif.latex?Q%28s%2Ca%3B%5Ctheta%2C%20%5Calpha%2C%20%5Cbeta%29%3DV%28s%3B%5Ctheta%2C%5Cbeta%29&plus;A%28s%2Ca%3B%5Ctheta%2C%5Calpha%29)
