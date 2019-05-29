这是自学的课程。

之前已知存在的问题就是看了又忘，总是记不住。觉得还是缺少一个高密度高频次的记忆过程，我不能奢望只是通过一次又一次地非持续地“阅读”来记住知识。要像准备考试一样，好好记一下。

------

参考书籍：《模式分类》

这是我所学的知识的大纲，其中有些符号的意义未做说明，若有不解请翻阅参考书籍

许多内容还没有完成

------

- 贝叶斯决策论

  - 基础

    - 贝叶斯公式

      -  $P(w_j|\mathbf x)=\frac{p(\mathbf x|w_j)P(w_j)}{p(\mathbf x)}=\frac{p(\mathbf x|w_j)P(w_j)}{\sum_{j=1}^2p(\mathbf x|w_j)P(w_j)}\qquad \mathbf x\mbox{ 为向量，}p\mbox{ 表示类条件概率密度,}P\mbox{ 表示概率}$
      - $posterior=\frac{likelihood\times prior}{evidence}\mbox{ 即：} 后验概率=\frac{似然函数\times 先验概率}{总概率}​$

    - 风险

      - 条件风险
        - $R(\alpha_i|\mathbf x)=\sum_{j=1}^c\lambda (\alpha_i|w_j)P(w_j|\mathbf x)\qquad \alpha_i为采取的行动，c为类别总数$
        - 选择$\alpha_i$使$R(\alpha_i|\mathbf x)$最小化，得到的风险为贝叶斯风险$R^*$
      - 总风险
        - $R=\int R(\alpha(\mathbf x)|\mathbf x) p(\mathbf x))d\mathbf x\qquad\mbox{在整个特征空间积分}$

    - 决策规则：有各种相互等价的形式

      - 最小误差决策规则
      - 最小风险决策规则

    - 极小化极大准则

      - 先验概率不确定或不知道时，需要尝试拜托先验概率

      - $R=\int_{R_1}[\lambda_{11}P(w_1)p(\mathbf x|w_1)+\lambda_{12}P(w_2)p(\mathbf x|w_2)]d\mathbf x+\int_{R_2}[\lambda_{21}P(w_1)p(\mathbf x|w_1)+\lambda_{22}P(w_2)p(\mathbf x|w_2)]d\mathbf x$

        由于 $P(w_1)=1-P(w_2)$，以及$\int_{R_1}p(\mathbf x|w_1)d\mathbf x=1-\int_{R_2}p(\mathbf x|w_1)d\mathbf x$

        $R(P(w_1))=\lambda_{22}+(\lambda_{12}-\lambda_{22})\int_{R_1}p(\mathbf x|w_2)d\mathbf x+P(w_1)[(\lambda_{11}-\lambda_{22})+(\lambda_{21}-\lambda_{11})\int_{R_2}p(\mathbf x|w_1)d\mathbf x-(\lambda_{12}-\lambda_{22})\int_{R_1}p(\mathbf x|w_2)d\mathbf x)]$

        令$P(w_1)$的系数为0就可以使风险和先验概率独立

        极小化极大风险$R_{mm}=\lambda_{22}+(\lambda_{12}-\lambda_{22})\int_{R_1}p(\mathbf x|w_2)d\mathbf x=\lambda_{11}+(\lambda_{21}-\lambda_{11})\int_{R_2}p(\mathbf x|w_1)d\mathbf x$

      - ”极小化极大“就体现在先验概率取任何值时所引起的总风险的最坏情况尽可能小

    - 判别函数$g_i(\mathbf x)$

      - 每一个判别函数都可以替换成$f(g_i(\mathbf x))$，其中$f(\cdot)$ 是一个单调递增函数，分类结果不变
      - 几个常见判别函数
        - $g_i(\mathbf x)=P(w_i|\mathbf x)=\frac{p(\mathbf x|w_i)P(w_i)}{\sum_{j=1}^cp(\mathbf x|w_j)P(w_j)}$
        - $g_i(x)=p(\mathbf x|w_i)P(w_i)$
        - $g_i(x)=ln\ p(\mathbf x|w_i)+lnP(w_i)$
      - 判别规则：$\forall j\ne i,g_j(\mathbf x)>g_i(\mathbf x),那么\mathbf x\rightarrow w_i$
      - 对两类情况取$g(\mathbf x)=g_1(\mathbf x)-g_2(\mathbf x)$

    - 正态密度

      - 单变量密度函数
        - $p(x)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2)\ \sim N(\mu,\sigma^2)$
        - 熵 $H(p(x))=-\int p(x)\ ln\ p(x)dx\quad 奈特(log_2时为比特)$   
        - 中心极限定理：大量小的独立的随机分布地总和等效为一个高斯分布
      - 多元密度函数
        - $p(\mathbf x)=\frac{1}{(2\pi)^{d/2}|\mathbf\Sigma|^{1/2}}exp(-\frac{1}{2}(\mathbf x-\mathbf\mu)^t\mathbf\Sigma^{-1}(\mathbf x-\mathbf\mu))\ \sim N(\mathbf\mu,\mathbf\Sigma)​$
        - 协方差矩阵$\Sigma$通常是对称半正定矩阵，这里我们严格限定其为正定矩阵
        - 服从正态分布地随机变量地线性组合也是正态分布
          - $p(\mathbf x)\sim N(\mathbf\mu,\mathbf\Sigma)\Rightarrow p(\mathbf y=\mathbf A^t\mathbf x)\sim N(\mathbf A^t\mathbf \mu,\mathbf A^t\mathbf\Sigma\mathbf A)$
          - 白化变换$\mathbf A_w=\mathbf\Phi\mathbf\Lambda^{-1/2}​$使得变换后的协方差矩阵为单位阵（$\mathbf\Phi​$的列向量为$\mathbf\Sigma​$的正交本征向量，$\mathbf\Lambda​$为相应本征值对应的对角矩阵）
        - 几何特征
          - 多元正态密度完全由$d+d(d+1)/2$个参数决定
          - 等密度点的轨迹为一超椭球体
            - $r^2=(\mathbf x-\mathbf\mu)^t\mathbf\Sigma^{-1}(\mathbf x-\mathbf\mu)\ (马氏距离)$
            - 超椭球体的体积（不要求）

  - 正态分布的贝叶斯判别：$g_i(\mathbf x)=-\frac{1}{2}(\mathbf x-\mathbf \mu_i)^t\mathbf\Sigma_i^{-1}(\mathbf x-\mathbf \mu_i)-\frac{d}{2}ln2\pi-\frac{1}{2}ln|\mathbf\Sigma_i|+lnP(w_i)$

    - $\mathbf\Sigma_i=\sigma^2\mathbf I$
      - 忽略掉与$i​$无关的常量，得到$g_i(\mathbf x)=\mathbf w_i^t\mathbf x+w_{i0}\quad 其中\mathbf w_i=\frac{1}{\sigma^2}\mathbf\mu_i,w_{i0}=\frac{-1}{2\sigma^2}\mathbf\mu_i^t\mathbf\mu_i+lnP(w_i)​$
      - 由$g_i(\mathbf x)=g_j(\mathbf x)$得到判定面：$\mathbf w^t(\mathbf x-\mathbf x_0)=0\quad 其中\mathbf w=\mathbf\mu_i-\mathbf\mu_j,\mathbf x_0=\frac{1}{2}(\mu_i+\mu_j)-\frac{\sigma^2}{||\mathbf\mu_i-\mathbf\mu_j||^2}ln\frac{P(w_i)}{P(w_j)}(\mathbf\mu_i-\mathbf\mu_j)$
        - 此方程定义了一个通过点$\mathbf x_0​$且与$\mathbf w​$正交的超平面
        - 点$\mathbf x_0​$在$\mathbf\mu_i,\mathbf\mu_j​$的连线上，且当$P(w_i)=P(w_j)​$时，位于中点处——最小距离分类器，若不相等，会向先验概率较小的那一方移动
    - $\mathbf\Sigma_i=\mathbf\Sigma$
      - 忽略掉与$i$无关的常量，得到$g_i(\mathbf x)=\mathbf w_i^t\mathbf x+w_{i0}\quad 其中\mathbf w_i=\mathbf\Sigma^{-1}\mathbf\mu_i,w_{i0}=-\frac{1}{2}\mathbf\mu_i^t\mathbf\Sigma^{-1}\mathbf\mu_i+lnP(w_i)​$
      - 由$g_i(\mathbf x)=g_j(\mathbf x)$得到判定面：$\mathbf w^t(\mathbf x-\mathbf x_0)=0\quad 其中\mathbf w=\mathbf\Sigma^{-1}(\mathbf\mu_i-\mathbf\mu_j),\mathbf x_0=\frac{1}{2}(\mu_i+\mu_j)-\frac{\sigma^2}{(\mathbf\mu_i-\mathbf\mu_j)^t\mathbf\Sigma^{-1}(\mathbf\mu_i-\mathbf\mu_j)}ln\frac{P(w_i)}{P(w_j)}(\mathbf\mu_i-\mathbf\mu_j)$
        - 此方程定义了一个通过点$\mathbf x_0$但不一定与$\mathbf w$正交的超平面
        - 点$\mathbf x_0$在$\mathbf\mu_i,\mathbf\mu_j$的连线上，且当$P(w_i)=P(w_j)$时，位于中点处，若不相等，会向先验概率较小的那一方移动（判定面甚至可以不落在两个均值向量之间）
    - $\mathbf\Sigma_i=任意$
      - 忽略掉与$i$无关的常量，得到$g_i(\mathbf x)=\mathbf x^t\mathbf W_i\mathbf x+\mathbf w_i^t\mathbf x+w_{i0}\quad 其中\mathbf W_i=-\frac{1}{2}\mathbf\Sigma_i^{-1},\mathbf w_i=\mathbf\Sigma_i^{-1}\mathbf\mu_i,w_{i0}=-\frac{1}{2}\mathbf\mu_i^t\mathbf\Sigma_i^{-1}\mathbf\mu_i-\frac{1}{2}ln|\mathbf\Sigma_i|+lnP(w_i)$
      - 判定面为超二次曲面

  - 误差

    - 利用积分求误差概率
      - $P(error)=\int_{R_2}p(\mathbf x|w_1)P(w_1)d\mathbf x+\int_{R_1}p(\mathbf x|w_2)P(w_2)d\mathbf x$（两类情况）
      - $P(correct)=\sum_{i=1}^c\int_{R_I}p(\mathbf x|w_i)P(w_i)d\mathbf x$（多类情况，出错的方式比正确多）
      - 通过最大化正确率或最小化错误率，可以得到贝叶斯误差率，**没有其他方法可以产生更小的误差率**（*<u>样本数量趋近于无穷极限时，神经网络产生一个相当于贝叶斯判别函数的最小二乘判别</u>*）
    - 正态密度的误差上界（*<u>定界的思想——精确计算不可行或太复杂时</u>*）
      - Chernoff界
      - Bhattacharyya界
      - 其他接近于高斯分布的分布，上两个界依然可用

  - 丢失特征或噪声特征：最基本的方法是尽可能多的恢复出内在的分布信息，然后使用贝叶斯判别规则        ~~公式还没有搞懂~~

    - 丢失特征可以看作噪声特征的极端情况
    - $P(w_i|\mathbf x_g,\mathbf x_b)=\frac{\int p(w_i|\mathbf x)p(\mathbf x)p(\mathbf x_b|\mathbf x_t)d\mathbf x_t}{\int p(\mathbf x)p(\mathbf x_b|\mathbf x_t)d\mathbf x_t}\quad 当p(\mathbf x_b|\mathbf x_t)在整个空间为1时，退化为丢失特征的情况$

  - 贝叶斯置信网

- 最大似然估计和贝叶斯参数估计

- 线性判别函数

- 支持向量机

  - 硬间隔超平面 $H=(\mathbf w,b)=\mathbf w\mathbf x+b$

    - 样本空间是线性可分的 **即** 一定能够找到一个最优超平面可以将不同的样本分开

    - 样本$\mathbf x$到超平面的距离为$\frac{|g(\mathbf x)|}{||\mathbf w||}$

    - 最优超平面：所有能够将样本正确分类的超平面中对新数据的分类错误率最小——*其实代表了它的泛化能力*

    - 对超平面中的权值$\mathbf w$和偏移$b$做任何正的尺度调整不会影响分类结果——为了便于比较不同的超平面（想想为什么尺度变换就可以比较不同的超平面了？）

      - $\mathbf w\leftarrow \frac{\mathbf w}{||\mathbf w||},b\leftarrow \frac{b}{||\mathbf w||}$

    - 分类规则：$\mathbf w\mathbf x+b\ge 1,正样本\quad \mathbf w\mathbf x+b\le-1,负样本$——这是一个约束条件而且需要先进行尺度变换

    - 两个边界超平面的间隔为$\frac{2}{||\mathbf w||}$，我们需要最大化这一间隔

    - 求解最优超平面的问题转化为：

      ​	$\min_{\mathbf w,b}\frac{1}{2}||\mathbf w||^2\qquad s.t.\ y_i(\mathbf w^t\mathbf x_i+b)\ge1,i=1,2,...,m​$

      - 拉格朗日乘子法——转化
      - SMO算法——求解
      - KKT条件——支持向量

  - 软间隔超平面

    - 引入松弛变量$\xi_i$处理非线性可分的情况

    - 目标函数-错误率：$\Theta(\xi)=\sum_{i=1}^mI(\xi_i-1)\le\sum_{i=1}^m\xi_i$

      - 原错误率非线性求解困难，运用**定界**的思想转化为线性函数

    - 求解最优超平面的问题转化为：

      ​	$\min_{\mathbf w,b}||\mathbf w||^2+C\sum_{i=1}^m\xi_i\qquad s.t.\ y_i(\mathbf w^t\mathbf x_i+b)\ge1-\xi_i且\xi_i\ge0,i=1,2,...,m​$

      - $\xi_i>1$则分错，$C$为正数，调整允许错分的样本数
      - 拉格朗日乘子法——转化
      - SMO算法——求解
      - KKT条件——支持向量

  - 支持向量机

    - 非线性可分问题通过升维变成线性可分问题

      - 问题：维数灾难和过拟合；计算需求大
      - SVM的解决：其泛化能力与间隔相关而与维数无关；核函数的引入避免了高维空间的运算

    - 核函数及其隐式映射

      - 隐式映射的概念
      - 三种常见核函数
        - 多项式核函数：$K(\mathbf x,\mathbf x')=(\mathbf x^T\mathbf x'+1)^p$
        - 径向基函数（RBF）：$K(\mathbf x,\mathbf x')=exp(-\frac{1}{2\sigma^2}||\mathbf x-\mathbf x'||^2)$
        - 两层感知器函数：$K(\mathbf x,\mathbf x')=tanh(\beta_0\mathbf x^T\mathbf x'+\beta_1)$

    - 求解最优超平面的问题转化为：

      ​	$\min_{\mathbf w,b}\frac{1}{2}||\mathbf w||^2\qquad s.t.\ y_i(\mathbf w^t\varphi(\mathbf x_i)+b)\ge1,i=1,2,...,m​$

      - 拉格朗日乘子法（最好把《凸优化》上的相关内容看一看）
      - SMO算法
      - 核函数
      - KKT条件

    - SMO算法

      - 二次规划问题

        ​	$\max J(\alpha)=1^T\alpha-\frac{1}{2}\alpha^TH\alpha,其中H_{ij}=y_iy_jK(x_i,x_j),约束条件：\sum_{i=1}^N\alpha_iy_i=0,0\le\alpha_i\le C,i=1,2,...,N$

      - 每次选择两个变量，固定其他变量迭代求解

- 多层神经网络
  - 引言
    - 线性判别函数处理非线性可分问题时，最主要的困难是选择合适的非线性$\varphi$函数
    - 神经网络是一种可以适应复杂模型的非常灵活的启发式的统计模式识别技术
    - 启发式的技巧
    - 网络的拓扑结构：对问题启发式的知识可以通过对隐含层的数目、节点单元个数、和反馈节点数目等的选择，而轻而易举地嵌入到网络结构中。
    - 正则化：选择或调整网络地复杂程度（置信风险+经验风险=结构风险）
  - 前馈运算与分类
    - $g_k(\mathbf x)\equiv z_k=f(\sum_{j=1}^{n_H}w_{kj}f(\sum_{i=1}^dw_{ji}x_i+w_{j0})+w_{k0})=f(\sum_{j=1}^{n_H}w_{kj}f(\sum_{i=0}^dw_{ji}x_i)+w_{k0})\quad x_0=1$
    - 只要给出足够数量的隐单元、适当的非线性函数和权值，任何从输入到输出的连续映射函数都可以用一个三层非线性网络实现（任何后验概率都可以用一个三层网络表示）
  - 反向传播算法（backpropagation）
    - $x_i\rightarrow y_j\rightarrow z_k\leftrightarrow t_k$ 
    - 目标函数：$J(\mathbf w)=\frac{1}{2}||\mathbf t-\mathbf z||^2$
    - $\Delta\mathbf w=-\eta\frac{\partial J}{\part\mathbf w}\quad \mathbf w(m+1)=\mathbf w(m)+\Delta\mathbf w$
    - 链式求导法则：
      - $\Delta w_{kj}=\eta\delta_ky_j\quad \delta_k=(t_k-z_k)f'(net_k)$
      - $\Delta w_{ji}=\eta\delta_jx_i\quad \delta_j=f'(net_j)\sum_{k=1}^cw_{kj}\delta_k$
      - 试着用矩阵来求导和表示。
      - 试着用更形象的方式去记忆和理解：若$y_j$和$(t_k-z_k)$都是正的且$f'(net_k)$一般也是正的，那么应该增大权值
    - 训练协议：随即训练(stochastic)、成批训练(batch)、在线训练(on-line)
      - epoch：一个epoch对应于训练集的所有样本都提供给输入层一次
    - 学习曲线：误差（目标函数）对训练总量（回合数）的函数
      - 训练集，测试集，验证集（交叉验证技术）
  - 误差曲面
    - 多重极小：不希望陷入有**较高的训练误差**的局部极小值
      - 较大型的网络：权值过剩可以帮助避免陷入局部极小值但同时带来过拟合的风险
      - 迭代梯度下降：初始化权值再训练一遍
      - 简单的启发式信息可以解决这个问题
  - 与贝叶斯理论的联系
    - $J_k(\mathbf w)=\sum_{\mathbf x}[g_k(\mathbf x;\mathbf w)-t_k]^2=\sum_{\mathbf x\in w_k}[g_k(\mathbf x;\mathbf w)-1]^2+\sum_{\mathbf x\notin w_k}[g_k(\mathbf x;\mathbf w)-0]^2=n\left\{\frac{n_k}{n}\frac{1}{n_k}\sum_{\mathbf x\in w_K}[g_k(\mathbf x;\mathbf w)-1]^2+\frac{n-n_k}{n}\frac{1}{n-n_k}\sum_{\mathbf x\notin w_k}[g_k(\mathbf x;\mathbf w)-0]^2\right\}$
    - $\tilde J(\mathbf w)\equiv\lim_{n\rightarrow \infty}\frac{1}{n}J_k(\mathbf w)=\int[g_k(\mathbf x;\mathbf w)-P(w_k|\mathbf x)]^2p(\mathbf x)d\mathbf x+\int P(w_k|\mathbf x)P(w_{i\neq k}|\mathbf x)p(\mathbf x)d\mathbf x$
      - 上式右边第二项与$\mathbf w$无关
      - 训练足够好的网络的输出将为：$g_k(\mathbf x;\mathbf w)\approx P(w_k|\mathbf x)$
      - 当样本数量趋近于无穷极限时，已训练过的网络的输出将可以近似成一个最小二乘意义上的后验概率（假设 这个网络可以表示后验概率函数）
    - 输出为概率：$z_k=\frac{e^{net_k}}{\sum_{m=1}^ce^{net_m}}$(softmax函数)
  - 正则化
    - 结构风险=经验风险+置信风险——稀疏性
    - 几种正则项
      - L0范数：权向量中非0的元素个数，难以求解优化
      - L1范数：权向量中各个元素的绝对值之和，L0的最优凸近似
      - L2范数（岭回归）：防止过拟合，帮助模型优化求解变得稳定和快速
  - 改进反向传播的一些实用技术
    - 激活函数
      - 期望性质
        - 非线性：提供更强的拟合能力
        - 饱和性：存在最大最小输出值，使得训练次数是有限的
          - 在分类网络中，当输出为概率时，饱和性很重要
          - 在回归网络中，并不太重要
        - 光滑性：导数存在
        - 单调性：有用但并非必要
        - 小输入时具有线性特性
      - 梯度消失（饱和）和梯度爆炸（怎么解决梯度爆炸？）
      - 分布式表示与全局性表示
    - sigmoid函数的参数
      - 最好使函数以0为中心并且反对称——更快的学习速率
      - $f(net)=\frac{2a}{1+e^{-bnet}}-a$  取$a=1.716,b=2/3$
    - 输入信号尺度变换
      - 为了避免“大数据”在训练网络时的支配作用，需先进行规格化，使每个特征的均值为0，方差相同（1.0）
    - 目标值
      - 如，对一个四类问题，模式属于$w_3\rightarrow \mathbf t=(-1,-1,1,-1)^t$
    - 带噪声的训练法
      - 数据增强
    - 人工“制造”数据
      - 数据增强
    - 隐单元数
      - 隐单元的个数决定了网络的表达能力
      - 过多的隐单元可能造成过拟合
    - 权值初始化
      - 快速而均衡（所有权值几乎同时达到最终的平衡值）的学习
      - 不能为0，否则学习不会开始；不能太大，否则在学习开始前就已经饱和；不能太小，否则只有线性网络被实现
    - 学习率
      - 经验：首先设为0.1，然后，学习过程中准则函数发散则减小之，学习速度过慢则增大之
    - 冲量项
      - 允许当误差曲面中存在平坦区时，网络可以以更快的速度学习
      - $\mathbf w(m+1)=\mathbf w(m)+(1-\alpha)\Delta\mathbf w_{bp}(m)+\alpha\Delta\mathbf w(m-1)$通常取$\alpha=0.9$
      - 平均化了随机权值更新，增加了稳定性；加快了学习过程，远离平坦区
    - 权值衰减
    - 线索
      - 增加一个输出单元来执行一个附加问题，该附加问题**不同于但又相关于**手头特定的分类问题
      - 线索信息比基于其他算法的分类器更易于嵌入到神经网络中
      - 比如，音素分类问题中，增加两个输出单元-元音和辅音。对于线索任务较为有用的特征很可能会促进类别的学习。
    - 在线训练、随机训练或成批训练
    - 停止训练
    - 隐含层数
      - 四层网络比三层网络更容易学习变换
      - 实验证明具有多个银行层的网络更易于陷入局部极小值中
    - 误差准则函数
      - 平方误差准则
      - 交叉熵准则 $J_{ce}(\mathbf w)=\sum_{m=1}^n\sum_{k-1}^ct_{mk}ln(t_{mk}/z_{mk})$
      - 闵科夫斯基误差
  - 几种网络