import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.io
import time
import os

import argparse

parser = argparse.ArgumentParser('Train or Test Arg! ')
# parser.add_argument('--step1', action='store_true')
# parser.add_argument('--step2', action='store_true')
# parser.add_argument('--step3', action='store_true')
# parser.add_argument('--step4', action='store_true')
parser.add_argument("--alpha", help="the vulue of alpha", type=float)
parser.add_argument("--l1depth", help="display a square of a given number", type=int)
parser.add_argument("--l1node", help="display a square of a given number", type=int)
parser.add_argument("--l2depth", help="display a square of a given number", type=int)
parser.add_argument("--l2node", help="display a square of a given number", type=int)
parser.add_argument("--roundtime", help="display a square of a given number", type=int)
# parser.add_argument("--prediction_time", help="display a square of a given number", type=int)
# python train.py --alpha 10 --l1depth 10 --l1node 50 --l2depth 4 --l2node 50 --roundtime 1


args = parser.parse_args()

# np.random.seed(1234)   
tf.set_random_seed(1234)

###############################################################################
############################## Helper Functions ###############################
###############################################################################


def plotfigure(t_idn, ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier, step):
    print(f'plotting for step%s'%step+'!!!')

    j = 1
    for y in [ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5]:
        plt.figure(j)
        plt.plot(t_idn, y,'--*b')  
        plt.plot(t_idn, u_pred_identifier[:,j-1],'--*r')  
        plt.savefig(f'./alpha%d/step%s_%d.jpg'%(args.alpha, step, j)) 
        j = j+1
    print(f'finished----plotting for step%s!!!'%step)


def Error_u(ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier, rroouunndd, sstteepp):
    error_u_id = [0, 0, 0, 0, 0, 0]
    j = 1
    for y in [ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5]:
        u_pred_id = u_pred_identifier[:,j-1].flatten()[:, None]
        error_u_id[j-1] = np.linalg.norm(y - u_pred_id, 2) / np.linalg.norm(y, 2)

        f = open("Error_test.txt", "a")    
        print(f'round:%d; setp:%d; Error u_%d: %e' % (rroouunndd, sstteepp, j, error_u_id[j-1]), file=f)
        j = j+1

    avg = np.average(error_u_id)
    f = open("Error_test.txt", "a")    
    print(f'round:%d; setp:%d;average average average Error u: %e' % (rroouunndd, sstteepp, avg), file=f)



def layerpstrudef(input_node, output_node, hidden_depth, hidden_node):
    layer = [input_node]
    for i in range(hidden_depth):
        layer.append(hidden_node)
    layer.append(output_node)
    return layer


def initialize_NN(layers,VarScope):
    weights = []
    biases = []
    num_layers = len(layers)
    with tf.variable_scope(VarScope):
        for l in range(0, num_layers - 1):
            W = xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
    return weights, biases


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)


def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


###############################################################################
################################ DeepHPM Class ################################
###############################################################################

class DeepHPM:
    def __init__(self,  u_layers, pde_layers, lb_idn, ub_idn):

        # Domain Boundary
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn

        # alpha 
        #self.alpha_value = alpha_value
        with tf.variable_scope("PDE"):
            self.beta_value = tf.Variable([0], dtype=tf.float32)
        self.alpha_value = 0.22 * tf.math.sigmoid(self.beta_value)
        # Init for Identification
        self.idn_init(u_layers, pde_layers)

        # tf session
        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        #                                              log_device_placement=False))

        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_)

        # config_ = tf.ConfigProto()
        # config_.gpu_options.per_process_gpu_memory_fraction = 0.15
        # self.sess = tf.Session(config=config_)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################

    def idn_init(self, u_layers, pde_layers):

        # Layers for Identification
        self.u_layers = u_layers
        self.pde_layers = pde_layers

        # Initialize NNs for Identification
        self.u_weights, self.u_biases = initialize_NN(u_layers, "Solution")
        self.pde_weights, self.pde_biases = initialize_NN(pde_layers, "PDE")

        self.Solution_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Solution")
        self.PDE_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "PDE")
        self.All_vars = tf.trainable_variables() 

        # tf placeholders for Identification
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.yt_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ALB_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.Cr_1_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.OSM_2_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ALT_3_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.TB_4_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.DB_5_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.u_tf = tf.concat([self.ALB_tf, self.Cr_1_tf, self.OSM_2_tf, self.ALT_3_tf, self.TB_4_tf, self.DB_5_tf],1)

        # tf graphs for Identification
        self.idn_u_pred, _ = self.idn_net_u(self.t_tf)
        # self.idn_u_pred, _ = self.idn_net_u(self.t_tf, self.yt_tf)
        self.idn_f_pred = self.idn_net_f(self.t_tf, self.yt_tf)

        # loss for Identification
        # step1
        self.idn_u_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.idn_u_pred - self.u_tf),1))

        # step2
        self.idn_f_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.idn_f_pred),1))
        
        # step3
        self.sol_u_pred, _ = self.idn_net_u(self.t_tf)
        self.sol_f_pred = self.sol_net_f(self.t_tf, self.yt_tf)

        # step4
        self.step4_pred = self.net_pde(self.u_tf)


        self.sol_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.sol_u_pred - self.u_tf),1))
        self.sol_loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.sol_f_pred),1))

        self.sol_loss = self.sol_loss_1 + self.sol_loss_2


        # Optimizer for Identification
        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_u_loss,
                                                                      var_list= self.Solution_vars, # self.u_weights + self.u_biases,
                                                                      method='L-BFGS-B',
                                                                      options={'maxiter': 50000,
                                                                               'maxfun': 50000,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol': 1.0 * np.finfo(float).eps})

        self.idn_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_f_loss,
                                                                      var_list= self.PDE_vars, # self.pde_weights + self.pde_biases + self.alpha_value,
                                                                      method='L-BFGS-B',
                                                                      options={'maxiter': 50000,
                                                                               'maxfun': 50000,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol': 1.0 * np.finfo(float).eps})

        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.sol_loss,
                                                                    var_list= self.All_vars, # self.u_weights + self.u_biases + self.alpha_value,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 50000,
                                                                             'maxfun': 50000,
                                                                             'maxcor': 50,
                                                                             'maxls': 50,
                                                                             'ftol': 1.0 * np.finfo(float).eps})


        self.idn_u_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_u_train_op_Adam = self.idn_u_optimizer_Adam.minimize(self.idn_u_loss, 
                                                                      var_list= self.Solution_vars) # self.u_weights + self.u_biases)

        self.idn_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(self.idn_f_loss,
                                                                      var_list= self.PDE_vars) # self.pde_weights + self.pde_biases + self.alpha_value)

        self.sol_optimizer_Adam = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.sol_optimizer_Adam.minimize(self.sol_loss,
                                                                  var_list= self.All_vars) # self.u_weights + self.u_biases + self.alpha_value)



    #for step1 & step2 & step3 
    def idn_net_u(self, t):
        X = t
        H = 2.0 * (X - self.lb_idn) / (self.ub_idn - self.lb_idn) - 1.0
        u = neural_net(H, self.u_weights, self.u_biases)
        u_t = tf.gradients(u, t)[0]

        return u, u_t


    def net_pde(self, terms):
        pde = neural_net(terms, self.pde_weights, self.pde_biases)

        return pde


    #step2 loss function
    def idn_net_f(self, t, yt):
        temp = int(24001*0.5)
        u, u_t = self.idn_net_u(t)
        zeros = tf.constant(0.0, shape=[temp,1])
        para = tf.concat([zeros+self.alpha_value, zeros, zeros, zeros, zeros, zeros], 1)
        # ttt =  tf.constant(self.alpha_value, shape=[temp,1])
        # para = tf.concat([ttt, zeros, zeros, zeros, zeros, zeros], 1)
        yt_array = tf.concat([yt, zeros, zeros, zeros, zeros, zeros], 1)
        yt_array_ = tf.multiply(para, yt_array)
        f =  u_t - (self.net_pde(u)+yt_array_)

        return f

    #step1 train function
    def idn_u_train(self, N_iter, t_, yt_, ALB_, Cr_1_, OSM_2_ ,ALT_3_, TB_4_, DB_5_):
        self.t = t_
        self.yt = yt_
        self.ALB = ALB_
        self.Cr_1 = Cr_1_
        self.OSM_2 = OSM_2_
        self.ALT_3 = ALT_3_
        self.TB_4 = TB_4_
        self.DB_5 = DB_5_

        tf_dict = {self.t_tf: self.t, self.yt_tf: self.yt, self.ALB_tf: self.ALB, self.Cr_1_tf: self.Cr_1, self.OSM_2_tf: self.OSM_2, self.ALT_3_tf: self.ALT_3, self.TB_4_tf: self.TB_4, self.DB_5_tf: self.DB_5}

        start_time = time.time()
        for it in range(N_iter):

            self.sess.run(self.idn_u_train_op_Adam, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_u_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        print('hello1! we now begin step1!!!!50%')

        self.idn_u_optimizer.minimize(self.sess,
                                      feed_dict=tf_dict,
                                      fetches=[self.idn_u_loss, self.alpha_value],
                                      loss_callback=self.callback)

        print('hello1! we now begin step1!!!!100%')


    #step2 train function
    def idn_f_train(self, N_iter, t_, yt_):
        self.t = t_
        self.yt = yt_

        tf_dict = {self.t_tf: self.t, self.yt_tf: self.yt}

        start_time = time.time()
        for it in range(N_iter):

            self.sess.run(self.idn_f_train_op_Adam, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_f_loss, tf_dict)
                trained_alpha_value = self.sess.run(self.alpha_value)

                print('It: %d, Loss: %.3e, Time: %.2f, alpha: %.5f' %
                      (it, loss_value, elapsed, trained_alpha_value))
                start_time = time.time()

        print('hello2! we now begin step2!!!!50%')

        self.idn_f_optimizer.minimize(self.sess,
                                      feed_dict=tf_dict,
                                      fetches=[self.idn_f_loss, self.alpha_value],
                                      loss_callback=self.callback)

        print('hello2! we now begin step2!!!!100%')


    #step1 step2 callback function
    def callback(self, loss, alphavalue):
        print('Loss: %e, alpha: %.5f' % (loss, alphavalue))


    #step1 prediction function
    def idn_predict(self, t_star, yt_star, ALB_test, Cr_1_test, OSM_2_test, ALT_3_test, TB_4_test, DB_5_test):

        tf_dict = {self.t_tf: t_star, self.yt_tf: yt_star, self.ALB_tf: ALB_test, self.Cr_1_tf: Cr_1_test, self.OSM_2_tf: OSM_2_test, self.ALT_3_tf: ALT_3_test, self.TB_4_tf: TB_4_test, self.DB_5_tf: DB_5_test}
        u_star = self.sess.run(self.idn_u_pred, tf_dict)

        return u_star


    #step3 loss_f function (which is the same as in step_2)  
    def sol_net_f(self, t, yt):
        temp = int(24001*0.5)
        u, u_t = self.idn_net_u(t)
        # u, u_t = self.idn_net_u(t, yt)
        zeros = tf.constant(0.0,shape=[temp,1])
        para = tf.concat([zeros+self.alpha_value, zeros, zeros, zeros, zeros, zeros], 1)
        # ttt =  tf.constant(0.1,shape=[temp,1])
        # para = tf.concat([ttt, zeros, zeros, zeros, zeros, zeros], 1)
        yt_array = tf.concat([yt, zeros, zeros, zeros, zeros, zeros], 1)
        yt_array_ = tf.multiply(para, yt_array)
        f =  u_t - (self.net_pde(u)+yt_array_)

        return f


    #step3 train function
    def sol_train(self, N_iter, t_, yt_, ALB_, Cr_1_, OSM_2_ ,ALT_3_, TB_4_, DB_5_):
        self.t = t_
        self.yt = yt_
        self.ALB = ALB_
        self.Cr_1 = Cr_1_
        self.OSM_2 = OSM_2_
        self.ALT_3 = ALT_3_
        self.TB_4 = TB_4_
        self.DB_5 = DB_5_

        tf_dict = {self.t_tf: self.t, self.yt_tf: self.yt, self.ALB_tf: self.ALB, self.Cr_1_tf: self.Cr_1, self.OSM_2_tf: self.OSM_2, self.ALT_3_tf: self.ALT_3, self.TB_4_tf: self.TB_4, self.DB_5_tf: self.DB_5}

        start_time = time.time()
        for it in range(N_iter):

            self.sess.run(self.sol_train_op_Adam, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                [loss_value, loss_value_1, loss_value_2] = self.sess.run([self.sol_loss, self.sol_loss_1, self.sol_loss_2], tf_dict)
                trained_alpha_value = self.sess.run(self.alpha_value)
                print('It: %d, : %.3e, Ls_1: %.3e, r_Ls_1: %.2f%%, Ls_2: %.3e, r_Ls_2: %.2f%%, Time: %.2f, alpha: %.5f' %
                      (it, loss_value, loss_value_1, 100*loss_value_1/loss_value, loss_value_2, 100*loss_value_2/loss_value, elapsed, trained_alpha_value))
                start_time = time.time()

        print('hello3! we now begin step3!!!!50%')


        self.sol_optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.sol_loss, self.sol_loss_1, self.sol_loss_2, self.alpha_value],
                                    loss_callback=self.callback_step_4)

        print('hello3! we now begin step1!!!!100%')
        

    #step3 callback function 
    def callback_step_4(self, loss, loss_1, loss_2, alphavalue):
        loss_value = loss
        loss_value_1 = loss_1
        loss_value_2 = loss_2
        aalphavalue = alphavalue
        print('Ls: %.3e, Ls_1: %.3e, r_Ls_1: %.2f%%, Ls_2: %.3e, r_Ls_2: %.2f%%, alpha: %.5f' %
              (loss_value, loss_value_1, 100*loss_value_1/loss_value, loss_value_2, 100*loss_value_2/loss_value, aalphavalue))


    #step3 prediction function
    def sol_predict(self, t_star, yt_star, ALB_test, Cr_1_test, OSM_2_test, ALT_3_test, TB_4_test, DB_5_test):

        tf_dict = {self.t_tf: t_star, self.yt_tf: yt_star, self.ALB_tf: ALB_test, self.Cr_1_tf: Cr_1_test, self.OSM_2_tf: OSM_2_test, self.ALT_3_tf: ALT_3_test, self.TB_4_tf: TB_4_test, self.DB_5_tf: DB_5_test}
        u_star = self.sess.run(self.idn_u_pred, tf_dict)

        return u_star

    #step4 predict
    def prediction_step4(self, x_0, x_1, x_2, x_3, x_4, x_5):
        tf_dict = {self.ALB_tf:x_0, self.Cr_1_tf:x_1, self.OSM_2_tf:x_2, self.ALT_3_tf:x_3, self.TB_4_tf:x_4, self.DB_5_tf:x_5}
        F_pred = self.sess.run(self.step4_pred, tf_dict)

        return F_pred 

    ###############################################################################
    ################################ Main Function ################################
    ###############################################################################

if __name__ == "__main__":

    # Doman bounds
    lb_idn = np.array([6.0])
    ub_idn = np.array([414.0])

    # Args_values
    alpha_value = 0.2 #args.alpha/100
    print(alpha_value)
    
    l1_hidden_depth = args.l1depth
    l1_hidden_node = args.l1node
    l2_hidden_depth = args.l2depth
    l2_hidden_node = args.l2node
    roundtime = args.roundtime

    for j in range(3,4):
        dirs = f'./roundtime%d/alpha%d/test_1_%d/'%(roundtime, args.alpha, j)
        if not os.path.exists(dirs):
            os.makedirs(dirs)


    # Layers
    u_layers = layerpstrudef(1, 6, l1_hidden_depth, l1_hidden_node)
    pde_layers = layerpstrudef(6, 6, l2_hidden_depth, l2_hidden_node)


    # Model
    model = DeepHPM(u_layers, pde_layers, lb_idn, ub_idn)


    ### Load Data ###
    data_idn = scipy.io.loadmat('./Data/data_cd_hptal.mat')
    t_idn = data_idn['t'].flatten()[:, None]
    yt_idn = data_idn['y_t'].flatten()[:, None]
    ALB = data_idn['ALB'].flatten()[:, None]
    Cr_1 = data_idn['Cr_1'].flatten()[:, None]
    OSM_2 = data_idn['OSM_2'].flatten()[:, None]
    ALT_3 = data_idn['ALT_3'].flatten()[:, None]
    TB_4 = data_idn['TB_4'].flatten()[:, None]
    DB_5 = data_idn['DB_5'].flatten()[:, None]

    t_idn =t_idn[0:24001 ,:]
    yt_idn =yt_idn[0:24001 ,:]
    ALB =ALB[0:24001 ,:]
    Cr_1 =Cr_1[0:24001 ,:]
    OSM_2 =OSM_2[0:24001 ,:]
    ALT_3 =ALT_3[0:24001 ,:]
    TB_4 =TB_4[0:24001 ,:]
    DB_5 =DB_5[0:24001 ,:]

    t_idn_star = t_idn
    yt_idn_star = yt_idn

    N_choose_datasets = int(24001 * 0.6)
    N_train=int(24001*0.5)
    N_test=int(24001*0.1)



    np.random.seed(roundtime)   

    idx = np.random.choice(t_idn_star.shape[0], N_choose_datasets, replace=False)

    t_train = t_idn_star[idx[0:N_train], :]
    # print(t_train.shape)
    # (18000, 1)
    yt_train = yt_idn[idx[0:N_train], :]
    ALB_train = ALB[idx[0:N_train], :]
    Cr_1_train = Cr_1[idx[0:N_train], :]
    OSM_2_train = OSM_2[idx[0:N_train], :]
    ALT_3_train = ALT_3[idx[0:N_train], :]
    TB_4_train = TB_4[idx[0:N_train], :]
    DB_5_train = DB_5[idx[0:N_train], :]


    t_test = t_idn_star[idx[N_train:N_choose_datasets], :]
    # print(t_train.shape)
    # (3600, 1)
    yt_test = yt_idn[idx[N_train:N_choose_datasets], :]
    ALB_test = ALB[idx[N_train:N_choose_datasets], :]
    Cr_1_test = Cr_1[idx[N_train:N_choose_datasets], :]
    OSM_2_test = OSM_2[idx[N_train:N_choose_datasets], :]
    ALT_3_test = ALT_3[idx[N_train:N_choose_datasets], :]
    TB_4_test = TB_4[idx[N_train:N_choose_datasets], :]
    DB_5_test = DB_5[idx[N_train:N_choose_datasets], :]


    if roundtime != 1:

        saver = tf.train.Saver()
        ckpt_last_round_step3 = f'./roundtime%d/alpha%d/test_1_3/test_1_3.ckpt'%(roundtime-1, args.alpha)
        save_path = saver.restore(model.sess, ckpt_last_round_step3)
        print(f'now it is the roundtime--%d, we reload ckpt_last_round_step3 successfully' % roundtime)

    else:
        saver = tf.train.Saver()
        checkpoint_file_step3_reload = './roundtime18/loadalpha10/test_1_3/test_1_3.ckpt'
        save_path = saver.save(model.sess, checkpoint_file_step3_reload)
        print(f'now it is the roundtime--%d, we reload ckpt_last_round_step3 successfully' % roundtime)


    # elif args.step3 :

    print('hello3! we now begin step3!333333333333333333')

    # saver = tf.train.Saver()
    # save_path = saver.restore(model.sess, checkpoint_file_step2)

    #step3 
    model.sol_train(N_iter=30000, t_=t_train, yt_=yt_train, ALB_=ALB_train, Cr_1_=Cr_1_train, OSM_2_=OSM_2_train ,ALT_3_=ALT_3_train, TB_4_=TB_4_train, DB_5_=DB_5_train)

    u_test_identifier = model.idn_predict(t_test, yt_test, ALB_test, Cr_1_test, OSM_2_test, ALT_3_test, TB_4_test, DB_5_test)

    #plotting for step3
    # plotfigure(t_idn, ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier, step=3)

    checkpoint_file_step3_save = f'./roundtime%d/alpha%d/test_1_3/test_1_3.ckpt'%(roundtime, args.alpha)

    saver = tf.train.Saver()
    save_path = saver.save(model.sess, checkpoint_file_step3_save)
    
    #Index output describing fitting degree
    Error_u(ALB_test, Cr_1_test, OSM_2_test, ALT_3_test, TB_4_test, DB_5_test, u_test_identifier, rroouunndd=roundtime, sstteepp=3)

    saver = tf.train.Saver()
    save_path = saver.save(model.sess, checkpoint_file_step3_save)    
    print('goodbye3! we have finished step3!333333333333333333')



