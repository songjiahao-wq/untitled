#新添加
try:
    from scipy.signal import savgol_filter
    import scipy.stats
    from filterpy.monte_carlo import multinomial_resample
except ImportError:
    pass
import numpy as np
##################
class Particle():
    def __init__(self):
        # 粒子活动范围
        self.x = np.zeros((2, 1))
        self.x_range=[0,960]
        self.y_range=[0,600]
        #粒子的初始运动方向和速度
        self.v_mean=0.006#0.0006
        self.v_std=0.004#0.00004
        #
        self.n_particles = 50#初始的粒子个数
        self.std_heading=3#-6
        self.std_v=5
        self.particles = self.create_particles()  # 初始化粒子状态
        self.weights = np.ones(self.n_particles) /self. n_particles  # 初始化权重

    def f(self, x):
        # State-transition function is identity
        return np.copy(x)

    def create_particles(self):
        """这里的粒子状态设置为（坐标x，坐标y，运动方向，运动速度）"""
        particles = np.empty((self. n_particles, 4))
        particles[:, 0] = np.random.uniform(self.x_range[0], self.x_range[1], size=self. n_particles)
        particles[:, 1] = np.random.uniform(self.y_range[0], self.y_range[1], size=self. n_particles)
        particles[:, 2] = np.random.uniform(0, 2 * np.pi, size=self.n_particles)
        particles[:, 3] = self.v_mean + (np.random.randn(self.n_particles) * self.v_std)
        return particles

    def predict_particles(self,particles):
        """这里的预测规则设置为：粒子根据各自的速度和方向（加噪声）进行运动，如果超出边界则随机改变方向再次尝试，"""
        idx = np.array([True] * len(particles))
        particles_last = np.copy(particles)
        for i in range(100):  # 最多尝试100次
            if i == 0:
                particles[idx, 2] = particles_last[idx, 2] + (np.random.randn(np.sum(idx)) * self.std_heading)
            else:
                particles[idx, 2] = np.random.uniform(0, 2 * np.pi, size=np.sum(idx))  # 随机改变方向
            particles[idx, 3] = particles_last[idx, 3] + (np.random.randn(np.sum(idx)) * self.std_v)
            particles[idx, 0] = particles_last[idx, 0] + np.cos(particles[idx, 2]) * particles[idx, 3]
            particles[idx, 1] = particles_last[idx, 1] + np.sin(particles[idx, 2]) * particles[idx, 3]
            # 判断超出边界的粒子
            idx = ((particles[:, 0] < self.x_range[0])
                   | (particles[:, 0] > self.x_range[1])
                   | (particles[:, 1] < self.y_range[0])
                   | (particles[:, 1] > self.y_range[1]))
            if np.sum(idx) == 0:
                break
        return particles

    def update_particles(self,particles, weights, z, d_std):
        """粒子更新，根据观测结果中得到的位置pdf信息来更新权重，这里简单地假设是真实位置到观测位置的距离为高斯分布"""
        # weights.fill(1.)
        distances = np.linalg.norm(particles[:, 0:2] - z, axis=1)  # 求范数
        weights *= scipy.stats.norm(0, d_std).pdf(distances)
        weights += 1.e-300
        weights /= sum(weights)
        return weights

    def estimate(self,particles, weights):
        """估计位置"""
        return np.average(particles, weights=weights, axis=0)  # 加权平均

    def neff(self,weights):
        """用来判断当前要不要进行重采样"""
        return 1. / np.sum(np.square(weights))

    def resample_from_index(self,particles, weights, indexes):
        """根据指定的样本进行重采样"""
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights /= np.sum(weights)
        return particles, weights

    def step(self, z):
        """迭代一次粒子滤波，返回状态估计"""
        #print('进入滤波函数')
        self.x=self.f(self.x)
        #particles=self.particles  # 初始化粒子状态
        #weights=self.weights
        # -5,-5//-0.6,-15
        #print('使用粒子滤波')
        for i in range(5):

            self.particles = self.predict_particles(self.particles)  # 1. 预测
            self.weights = self.update_particles(self.particles, self.weights, z, 2)  # 2. 更新
            if self.neff(self.weights) < len(self.particles) / 2:  # 3. 重采样
                indexes = multinomial_resample(self.weights)
                self.particles, self.weights = self.resample_from_index(self.particles, self.weights, indexes)
            # estimate(particles, weights)
            self.x=self.estimate(self.particles, self.weights)
        return self.x

class Savgol():
    def __init__(self):
        self.x = np.zeros((2, 1))
        #savgol_filter

    def step(self,posi):
        """
        input:输入数据的shape:[视频总帧数,关键点的类别数,关键点的坐标维度]
        """
        posi1=np.array(posi)
        result_mat = np.zeros_like(posi1)
        #posi (126, 24, 2)
        for i in range(len(posi[0])):
            #print(len(posi[0]))
            kpt=[posi1[:,i,:]]
            kpt=np.array(kpt)
            kpt=kpt.transpose(0,2,1)
            """"savgol滤波器常用参数：x:要过滤的数据。window_length:滤波器窗口的长度,值必须是单数，它越大，则平滑效果越明显；越小，则更贴近原始曲线。
                polyorder：用于拟合样本的多项式的阶数，它越小，则平滑效果越明显；越大，则更贴近原始曲线。
            """
            result=savgol_filter(kpt,7,1).transpose(2,1,0)
            #result=result[:,:,0]
            result_mat[:,i,:]=result[:,:,0]

        return list(result_mat)