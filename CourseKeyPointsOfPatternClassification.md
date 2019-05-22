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