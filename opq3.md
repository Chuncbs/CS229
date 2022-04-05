**3.1 A Non-Parametric Solution**

步骤（i）固定R最优化$\{C^m\}_m^{M}$
       定义$\hat{x}=Rx$且$\hat{c}=Rc$，因为$R$是正交矩阵所以满足$||x-c||^2=||\hat{x}-\hat{c}||^2$。当R固定时优化变为：
       $$\underset{C^1,\cdots C^M}{min}\sum_{\hat{x}}||\hat{x}-\hat{c}(i(\hat{x}))||^2\\s.t.\,\,\hat{c}\in \hat{C^1}\times \cdots \hat{C^M}$$

步骤（i i）固定$\{C^m\}_m^{M}$最优化$R$

 $$\underset{R}{min}\sum_{\hat{x}}||Rx-\hat{c}(i(\hat{x}))||^2\\s.t.\,\,R^TR=I$$
 
 在步骤2中$\hat{c}(i(\hat{x}))$是固定的，它是各个子空间子codeword的级连联。定义$\hat{c}(i(\hat{x}))$为$y$,对于n个训练样本，定义$X$和$Y$分别为$D\times n$的举例其列向量分别为$x$和$y$。步骤二可以写为：
 $$\underset{R}{min}\sum_{\hat{x}}||X-Y||_{F}^2\\s.t.\,\,R^TR=I$$
其中$||.||_F$是 Frobenius 范数。上述优化问题叫做正交普鲁克问题（Orthogonal Procrustes problem）其最优解为的近似解为：对$XY^T$进行SVD分解$XY^T=USV^T$，$R=VU^T$

算法解步骤为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/df02eee6be75450d9af5383811b6480a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBATGlnaHRfYmx1ZV9sb3Zl,size_20,color_FFFFFF,t_70,g_se,x_16)
在实际使用中大约迭代100次就可以有不错的效果，这个算法和其他常见的迭代算法一样容易陷入局部最优的情况，初始化对结果又着比较大影响。

**3.2 A Parametric Solution**

该解法基于数据服从高斯分布的假设，该参数解既有实践意义又有理论意义。第一如果数据服从高斯分布那么这个方法可以提供一个更为简单的解答方法。第二这个方法也可以用来初始话方法1。

对于OPQ,优化目标的挑战是既要优化$R$也要优化$\{C^m\}_m^{M}$。当前基于数据服从高斯分布假设，量化失真的下界是依赖$R$而非$\{C^m\}_m^{M}$，因此允许我们直接优化$R$.

高斯分布假设只是用来做理论推导用的，实际使用该方法不需要依赖这些假设。

3.2.1 Distortion Bound of Quantization
我们假设$X\sim \mathcal{N}(0,\Sigma)$的高斯分布，根据信息论中的失真率理论(rate distortion theory)，量化失真的期望满足

$$E\ge k^{-\frac{D}{2}}D|\Sigma|^{\frac{1}{D}}\tag{8}$$。

上面不等式给出了针对任何量化方法使用$k$个codeword的失真下界。

下面表格显示了使用Kmeans方法在$10^5$个样本,$k=256$,$\sigma^2$在$[0.5,1]$之间随机生成得的期望和理论边界对比。从下表可以看出使用公式8作为量化失真的下界是合理的，之所以期望比理论值高是因为可能Kmeans遇到了局部最优和固定编码长度的影响。
![在这里插入图片描述](https://img-blog.csdnimg.cn/93816332268a49c6a6b001971981a36d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBATGlnaHRfYmx1ZV9sb3Zl,size_20,color_FFFFFF,t_70,g_se,x_16)

**3.2.2 Distortion Bound of Product Quantization**
基于$X\sim \mathcal{N}(0,\Sigma)$假设，$\hat{x}=Rx\sim\mathcal{N}(0,\hat{\Sigma})$其中$\hat{\Sigma}=R\Sigma R^T$。可以将矩阵$\hat{\Sigma}$分解为：
$$\hat{\Sigma}=\begin{bmatrix}\hat{\Sigma_{11}}&\cdots&\hat{\Sigma_{1M}}\\\vdots&\ddots&\vdots\\\hat{\Sigma_{M1}}&\cdots&\hat{\Sigma_{MM}}\end{bmatrix}\tag{9}$$

其中对角子矩阵$\hat{\Sigma}_{mm}$是第$m$个子空间的协方差矩阵，$\hat{x}^{m}\sim \mathcal{N}(0,\hat{\Sigma}_{mm})$,对于第$m$子空间其失真不小于$k^{-\frac{D}{2M}}\frac{D}{M}|\Sigma|^{\frac{M}{D}}$,所以PQ的量化失真满足：
$$E(R)\ge k^{-\frac{D}{2M}}\frac{D}{M}\sum_{m=1}^{M}|\hat{\Sigma}_{mm}|^{\frac{M}{D}}\tag{10}$$	

从式子10可知量化失真的下界不依赖于码本。

**3.2.3 Minimizing the Distortion Bound**

如果下界和期望值足够接近，那么我们可以通过最小化下界，进而达到最小化量化失真的目的。因此提出通过优化$R$最小化下界。

$$
\begin{cases}\underset{R}{min}\sum_{m=1}^{M}|\hat{\Sigma}_{mm}|^{\frac{M}{D}}\\
s.t. \,\,R^TR=I\end{cases}\tag{11}
$$

上式为优化一个函数基于正交限制，一般这类问题是一个非凸问题，因此很难在有限计算内求解。我们发现11中的目标函数存在与R无关的下界。这个下界成立需要一个较为温和的假设，因此优化11式中的目标函数达到了最低下界。
>We find the objective in (11) has a constant lower bound
independent of R. This lower bound is achievable under a
very mild assumption. As a result, optimizing the objective
in (11) is equivalent to achieving its lower bound.

使用算数几何平均不等式（AMGM）有：
$$\sum_{m=1}^{M}|\hat{\Sigma}_{mm}|^{\frac{M}{D}}\ge M\prod_{m=1}^{M}|\hat{\Sigma}_{mm}|^{\frac{1}{D}}\tag{12}$$

式子12成立的条件为各个$|\hat{\Sigma}_{mm}|$相等，进一步使用Fischer’s 不等式有：
$$\prod_{i=1}^{M}|\hat{\Sigma}_{mm}|\ge|\hat{\Sigma}|\tag{13}$$

上式等式成立当且仅当在$\hat{\Sigma}$中非对角子矩阵均为0矩阵，因为$\hat{\Sigma}=R\Sigma R^T$因此有$|\hat{\Sigma}|=|\Sigma|$

根据式子12,13，量化失真的下界有：

$$\sum_{m=1}^{M}|\hat{\Sigma}_{mm}|^{\frac{M}{D}}\ge M|\Sigma|^{\frac{1}{D}}\tag{14}$$

上式当12，13同时满足时14等式成立。

 1. 独立性：使用PCA处理数据可以使得式13成立,通过PCA处理后各个维度的协方差为0。各个子空间项目独立
 2. 平衡各个子空间的方差，如果各个$|\hat{\Sigma}_{mm}|$相等那么式12成立。通过PCA处理数据后，$|\Sigma_{mm}|$等于$\Sigma_{mm}$奇异值的乘积。我们可以重新排列各个主成分进而平衡各个各个子空间的奇异值乘积使得个各个$|\hat{\Sigma}_{mm}|$相等（温和假设）。进而12，13成立，量化失真的下界可以达到最小。

**3.2.4 Algorithm: Eigenvalue Allocation**

---

**3.2.4 Algorithm: Eigenvalue Allocation**
上面提出一个奇异值分配优化式子11的方法。

通过PCA处理数据，降序排列奇异值$\sigma_1^2\ge\sigma_2^2\ge\cdots\sigma_D^2$,准备M个桶，挑选出最大奇异值分配给拥有奇异值之和最小的桶(除非通已将装满，桶容量为$\frac{D}{M}$),这相当于将各个奇异值分配给各个子空间,形成各个子空间的主成分，算法通过重新排列各个奇异值向量形成R

在真实数据 SIFT1M/GIST1M上验证，奇异值分配算法表现足够好，下表左侧展示理论最小值，右侧为使用奇异值分配得到的目标函数值。
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe200a440a2f4e5996ce96ccf71aff71.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBATGlnaHRfYmx1ZV9sb3Zl,size_20,color_FFFFFF,t_70,g_se,x_16)
参数解决方案的总结：首先计算数据$D\times D$大小的协方法矩阵，然后使用奇异值分配方法来生成R，再在RX上使用PQ算法。

---

























