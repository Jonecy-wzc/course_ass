import numpy as np
import matplotlib.pyplot as plt 
import math

def generate_data(mean_list, covariance_list, num_list):
    np.random.seed(0)
    data = np.random.multivariate_normal(mean_list[0], covariance_list[0], num_list[0])
    for i in range(len(mean_list)-1):
        new_data = np.random.multivariate_normal(mean_list[i+1], covariance_list[i+1], num_list[i+1])
        data = np.append(data, new_data, 0)
    return data

def distribution_plot(data, num_list):
    color_list = ['hotpink', 'slateblue', 'aquamarine']
    x = data[:,:,0]; y = data[:,:,1]
    l = 0; r = num_list[0] - 1
    plt.figure(figsize=(16,9))
    plt.scatter(x[l:r], y[l:r], c = color_list[0])
    for i in range(len(num_list)- 1):
        l += num_list[i]; r += num_list[i+1]
        plt.scatter(x[l:r], y[l:r], c = color_list[i+1])
    plt.title('The distribution of data generated by Gaussian', fontdict={'fontsize': 20})
    plt.show()

class Gaussian_wzc:
    def __init__(self, data, k):
        self.k = k
        self.x = data

    # P(x_i | mu, epsilon)
    def marginal_P1(self, mu, epsilon, i):
        result = np.zeros((self.k,1))
        dim = 2
        for j in range(self.k):
            xdiff = (self.x[i] - mu[j]).reshape((1,dim))
            #print(np.linalg.inv(epsilon[j]+0.001*np.eye(dim)))
            prob = 1.0/np.power(2*np.pi,dim/2)*np.power(np.abs( np.linalg.det(epsilon[j]) ),0.5)*np.exp(-0.5*xdiff.dot(np.linalg.inv(epsilon[j]+0.001*np.eye(dim))).dot(xdiff.T))
            result[j] = prob
        return result                                                    #返回一个数组(3,1)

    # P(x_i)   和的形式
    def marginal_Pt(self, alpha_list, marginal_P_list_single):   #marginal_P_list_single = 3*1
        result = 0
        for i in range(self.k):
            result += alpha_list[i] * marginal_P_list_single[i]    
        return result                                            #返回一个数组 (1,)！

    #计算单个gamma_i  (gamma_list)
    def gamma_i(self, alpha_list, marginal_P_list, i):           #marginal_P_list = 900*3*1
        result = np.zeros((self.k,1))
        for j in range(self.k):
            molecular = alpha_list[j]*marginal_P_list[i][j]                   #alpha_list = 3*1 ; molecular = 1 * 1
            #print(alpha_list[j]*marginal_P_list[i][j])
            demoninator = self.marginal_Pt(alpha_list, marginal_P_list[i])                #输入marginal_P_list[i] = 3*1
            result[j] = molecular/demoninator
        return result             #返回一个数组 (3,1)

    #计算所有gamma和
    def sum_gamma(self, gamma_list):                             #gamma_list = 900*3*1
        sum = np.zeros(gamma_list[0].shape)                      #sum = 3*1
        n = len(gamma_list)                                      #n = 900                 
        for i in range(n):
            sum += gamma_list[i]             
        return sum                                               #返回一个数组 (3,1)     

    def mu_iter(self, x, gamma_list, sum_gamma):
        n = len(x); k = self.k                           #n = 900 ; k = 3
        a, b = x[0].shape                                        #a = 1, b = 2
        molecular = np.zeros((k,a,b))                            #molecular = 3*1*2
        demoninator = sum_gamma                                  #demoinator = 3*1
        for j in range(k):
            for i in range(n):
                molecular[j] += gamma_list[i][j]*x[i]            #1*1*2=1*2
            molecular[j] = molecular[j]/demoninator[j]           #1*2/1
            #print(demoninator)
        return molecular                                         #3*1*2

    def epsilon_iter(self, x, mu, gamma_list, sum_gamma):
        m = x[0].shape[1]                                        #m = 2
        k = self.k                                               #k = 3
        result = np.zeros((k,m,m))                               #result = 2*2
        for j in range(k):
            for i in range(len(x)):                              #2*1 * 1*2  = 2*2 
                result[j] +=  gamma_list[i][j]*np.dot((x[i]-mu[j]).T, (x[i]-mu[j]))    
            result[j] = result[j]/sum_gamma[j]   
        return result                                            #返回3*2*2的数组

    def alpha_iter(self, sum_gamma, n):
        k = self.k                                       # k = 3
        result = np.zeros((k,1))
        for i in range(k):
            result[i] = sum_gamma[i]/n                           #result = 3*1
        return result                             

    def LL(self, data, mean, var, alpha):
        dim = mean.shape[2]
        N = len(data)
        log_prob_list = []
        for i in range(N):
            prob_list = []
            for j in range(len(mean)):
                xdiff = (data[i] - mean[j])
                prob = 1.0/np.power(2*np.pi,dim/2)*np.power(np.abs( np.linalg.det(var[j]) ),0.5)*np.exp(-0.5*xdiff.dot(np.linalg.inv(var[j]+np.eye(dim)*0.001)).dot(xdiff.T)) 
                prob_list.append(alpha[j]*prob)
            temp = math.log(np.sum(prob_list))
            log_prob_list.append(temp)
        #Mean = np.array(log_prob_list).mean()
        Sum = np.array(np.sum(log_prob_list))
        return Sum

    def Gaussian_EM(self):
        n = len(self.x)                                                     # n = 900
        count = 0
        '''赋初始值'''
        alpha_list = np.ones((self.k,1))/self.k                             # 3*1   权值 = 1/k
        epsilon = np.array([(np.eye(2)*0.1 + 0.1) for i in range(self.k)])        # 3*2*2 对角单位阵
        mu = np.array([[4,1],[6,8],[10,2]]).reshape((3,-1,2))              # 3*1*2
        
        new_ll = self.LL(self.x, mu, epsilon, alpha_list)
        old_ll = new_ll + 1

        '''开始EM算法迭代'''
        while(np.abs(old_ll - new_ll) > 0.000001):
            old_ll = new_ll
            '''E步'''
            '''计算marginal_P_list'''
            marginal_P_list = []
            for i in range(n):
                marginal_P_list.append(self.marginal_P1(mu, epsilon, i))
            marginal_P_list = np.array(marginal_P_list)                     #转成ndarray类型  900*3*1

            '''计算gamma_list'''
            gamma_list = []
            for i in range(n):
                gamma_list.append(self.gamma_i(alpha_list, marginal_P_list, i)) 
            gamma_list = np.array(gamma_list)                               #转成ndarray类型  900*3*1

            '''M步，更新参数'''
            sum_gamma = self.sum_gamma(gamma_list)
            mu = self.mu_iter(self.x, gamma_list, sum_gamma)
            epsilon = self.epsilon_iter(self.x, mu, gamma_list, sum_gamma)               
            alpha_list = self.alpha_iter(sum_gamma, n)

            '''计算目标函数'''
            new_ll = self.LL(self.x, mu, epsilon, alpha_list)
            count += 1
        print('Convergences after %d iters!'%(count + 1))
        return mu, epsilon, alpha_list

def main():
    '''生成数据用到的参数'''
    mean_list = [[3,1],[8,10],[12,2]]
    cov_list = [[[1,-0.5],[-0.5,1]],[[2,0.8],[0.8,2]],[[1,0],[0,1]]]
    num_list = [300,300,300]
    # k = len(mean_list)
    k = 3
    '''生成数据'''
    data = generate_data(mean_list, cov_list, num_list)
    data = data.reshape((900,-1,2))

    '''绘制初始分布图像'''                   
    # distribution_plot(data, num_list)

    '''构造Gaussian类'''
    gaus = Gaussian_wzc(data, k)   

    '''计算聚类后的分类信息'''
    mu, epsilon, alpha_list = gaus.Gaussian_EM()

    '''打印结果'''
    print('\nThe final mean: \n',mu)
    print('\nThe final covariance: \n',epsilon)
    print('\nThe final weights: \n',alpha_list)

    # '''比较结果'''
    # mean_list = np.array(mean_list).reshape((3,-1,2))
    # cov_list = np.array([[[1,-0.5],[-0.5,1]],[[2,0.8],[0.8,2]],[[1,0],[0,1]]])
    # print('\nThe aberration of mean: \n',mu-mean_list)
    # print('\nThe aberration of covariance: \n',epsilon-cov_list)
    # print('\nThe aberration of weights: \n',alpha_list-1/3)

    '''计算AIC,BIC'''
    dim = 2 ; n = 900
    logL = gaus.LL(data, mu, epsilon, alpha_list)
    phi = k*dim + k*(1+dim)*dim/2 + k - 1
    Aic = -2*logL + 2*phi
    Bic = -2*logL + phi*math.log(n)
    print('AIC：%.2f \nBIC：%.2f'%(Aic,Bic))

if __name__ == "__main__":
    main()