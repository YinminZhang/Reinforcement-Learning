# Reinforcement Learning Algorithm

## Value based
- [Q Learning](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/Q_learning)
$$q^{T}(s,a)=q^{T-1}(s,a) +\frac 1 N[r(s')+ \max_{a'}q^{T-1}(s',a')-q^{T-1}(s,a)]$$
- [Sarsa](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/sarsa)
$$q^{T}(s,a)=q^{T-1}(s,a) +\frac 1 N[r(s')+ q^{T-1}(s',a')-q^{T-1}(s,a)]$$
- [Sarsa Lambda](https://github.com/YisraelZhang/Reinforcement-Learning/tree/master/sarsa_lambda)
$$q^{T}(s,a)=q^{T-1}(s,a) +\frac 1 N[r(s')+ q^{T-1}(s',a')-q^{T-1}(s,a)]E(s,a)$$
$E(s,a)$ is eligibility trace.