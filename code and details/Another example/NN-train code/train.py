# python ./alpha12/train.py --step? --alpha 12 --l1depth 10 --l1node 50 --l2depth 4 --l2node 50


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.io
import time

import argparse

parser = argparse.ArgumentParser('Train or Test Arg! ')
parser.add_argument('--step1', action='store_true')
parser.add_argument('--step2', action='store_true')
parser.add_argument('--step3', action='store_true')
parser.add_argument('--step4', action='store_true')
parser.add_argument("--alpha", help="the vulue of alpha", type=float)
parser.add_argument("--l1depth", help="display a square of a given number", type=int)
parser.add_argument("--l1node", help="display a square of a given number", type=int)
parser.add_argument("--l2depth", help="display a square of a given number", type=int)
parser.add_argument("--l2node", help="display a square of a given number", type=int)
# parser.add_argument("--prediction_time", help="display a square of a given number", type=int)



args = parser.parse_args()

np.random.seed(1234)   
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


def Error_u(ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier):

    j = 1
    for y in [ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5]:
        u_pred_id = u_pred_identifier[:,j-1].flatten()[:, None]
        error_u_id = np.linalg.norm(y - u_pred_id, 2) / np.linalg.norm(y, 2)
        print(f'Error u_%d: %e' % (j, error_u_id))
        j = j+1


def layerpstrudef(input_node, output_node, hidden_depth, hidden_node):
    layer = [input_node]
    for i in range(hidden_depth):
        layer.append(hidden_node)
    layer.append(output_node)
    return layer


def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers)
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
    def __init__(self,  u_layers, pde_layers, lb_idn, ub_idn, lb_sol, ub_sol, alpha_value):

        # Domain Boundary
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn

        self.lb_sol = lb_sol
        self.ub_sol = ub_sol

        # alpha 
        self.alpha_value = alpha_value

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
        self.u_weights, self.u_biases = initialize_NN(u_layers)
        self.pde_weights, self.pde_biases = initialize_NN(pde_layers)

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
        # self.sol_u_pred, _ = self.idn_net_u(self.t_tf, self.yt_tf)
        self.sol_f_pred = self.sol_net_f(self.t_tf, self.yt_tf)

        # step4
        self.step4_pred = self.net_pde(self.u_tf)


        self.sol_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.sol_u_pred - self.u_tf),1))
        self.sol_loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.sol_f_pred),1))

        self.sol_loss = self.sol_loss_1 + self.sol_loss_2


        # Optimizer for Identification
        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_u_loss,
                                                                      var_list=self.u_weights + self.u_biases,
                                                                      method='L-BFGS-B',
                                                                      options={'maxiter': 50000,
                                                                               'maxfun': 50000,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol': 1.0 * np.finfo(float).eps})

        self.idn_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_f_loss,
                                                                      var_list=self.pde_weights + self.pde_biases,
                                                                      method='L-BFGS-B',
                                                                      options={'maxiter': 50000,
                                                                               'maxfun': 50000,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol': 1.0 * np.finfo(float).eps})

        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.sol_loss,
                                                                    var_list=self.u_weights + self.u_biases,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 50000,
                                                                             'maxfun': 50000,
                                                                             'maxcor': 50,
                                                                             'maxls': 50,
                                                                             'ftol': 1.0 * np.finfo(float).eps})


        self.idn_u_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_u_train_op_Adam = self.idn_u_optimizer_Adam.minimize(self.idn_u_loss,
                                                                      var_list=self.u_weights + self.u_biases)

        self.idn_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(self.idn_f_loss,
                                                                      var_list=self.pde_weights + self.pde_biases)

        self.sol_optimizer_Adam = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.sol_optimizer_Adam.minimize(self.sol_loss,
                                                                  var_list=self.u_weights + self.u_biases)



    #for step1 & step2 & step3 
    def idn_net_u(self, t):
        # X = tf.concat([t], 1)
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
        u, u_t = self.idn_net_u(t)
        # u, u_t = self.idn_net_u(t, yt)
        zeros = tf.constant(0.0, shape=[21801,1])
        ttt =  tf.constant(self.alpha_value, shape=[21801,1])
        para = tf.concat([ttt, zeros, zeros, zeros, zeros, zeros], 1)
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
                                      fetches=[self.idn_u_loss],
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
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        print('hello2! we now begin step2!!!!50%')

        self.idn_f_optimizer.minimize(self.sess,
                                      feed_dict=tf_dict,
                                      fetches=[self.idn_f_loss],
                                      loss_callback=self.callback)

        print('hello2! we now begin step2!!!!100%')


    #step1 step2 callback function
    def callback(self, loss):
        print('Loss: %e' % (loss))

    #step1 prediction function
    def idn_predict(self, t_star, yt_star):
        tf_dict = {self.t_tf: t_star, self.yt_tf: yt_star}
        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        f_star = self.sess.run(self.idn_f_pred, tf_dict)

        return u_star, f_star


    #step3 loss_f function (which is the same as in step_2)  
    def sol_net_f(self, t, yt):
        u, u_t = self.idn_net_u(t)
        # u, u_t = self.idn_net_u(t, yt)
        zeros = tf.constant(0.0,shape=[21801,1])
        ttt =  tf.constant(0.1,shape=[21801,1])
        para = tf.concat([ttt, zeros, zeros, zeros, zeros, zeros], 1)
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
                print('It: %d, : %.3e, Ls_1: %.3e, r_Ls_1: %.2f%%, Ls_2: %.3e, r_Ls_2: %.2f%%, Time: %.2f' %
                      (it, loss_value, loss_value_1, 100*loss_value_1/loss_value, loss_value_2, 100*loss_value_2/loss_value, elapsed))
                start_time = time.time()

        print('hello3! we now begin step3!!!!50%')


        self.sol_optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.sol_loss, self.sol_loss_1, self.sol_loss_2],
                                    loss_callback=self.callback_step_4)

        print('hello3! we now begin step1!!!!100%')
        

    #step3 callback function 
    def callback_step_4(self, loss, loss_1, loss_2):
        loss_value = loss
        loss_value_1 = loss_1
        loss_value_2 = loss_2
        print('Ls: %.3e, Ls_1: %.3e, r_Ls_1: %.2f%%, Ls_2: %.3e, r_Ls_2: %.2f%%' %
              (loss_value, loss_value_1, 100*loss_value_1/loss_value, loss_value_2, 100*loss_value_2/loss_value))


    #step3 predict
    def sol_predict(self, t_star, yt_star):
        tf_dict = {self.t_tf: t_star, self.yt_tf: yt_star}
        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        f_star = self.sess.run(self.idn_f_pred, tf_dict)

        return u_star, f_star


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
    ub_idn = np.array([248.0])

    lb_sol = np.array([6.0])
    ub_sol = np.array([248.0])


    checkpoint_file_step1 = f'./alpha%d/test_1_1/test_1_1.ckpt'%args.alpha
    checkpoint_file_step2 = f'./alpha%d/test_1_2/test_1_2.ckpt'%args.alpha
    checkpoint_file_step3 = f'./alpha%d/test_1_3/test_1_3.ckpt'%args.alpha

    # Args_values
    alpha_value = args.alpha/100
    print(alpha_value)
    
    l1_hidden_depth = args.l1depth
    l1_hidden_node = args.l1node
    l2_hidden_depth = args.l2depth
    l2_hidden_node = args.l2node


    # Layers
    u_layers = layerpstrudef(1, 6, l1_hidden_depth, l1_hidden_node)
    pde_layers = layerpstrudef(6, 6, l2_hidden_depth, l2_hidden_node)


    # Model
    model = DeepHPM(u_layers, pde_layers, lb_idn, ub_idn, lb_sol, ub_sol, alpha_value)

    ### Load Data ###
    data_idn = scipy.io.loadmat('./Data/data_cd_hptal.mat')
    t_idn = data_idn['tt'].flatten()[:, None]
    yt_idn = data_idn['yy'].flatten()[:, None]
    ALB = data_idn['ALB'].flatten()[:, None]
    Cr_1 = data_idn['Cr'].flatten()[:, None]
    OSM_2 = data_idn['OSM'].flatten()[:, None]
    ALT_3 = data_idn['ALT'].flatten()[:, None]
    TB_4 = data_idn['TB'].flatten()[:, None]
    DB_5 = data_idn['DB'].flatten()[:, None]

    t_idn=t_idn[0:21801,:]
    yt_idn=yt_idn[0:21801,:]
    ALB=ALB[0:21801,:]
    Cr_1=Cr_1[0:21801,:]
    OSM_2=OSM_2[0:21801,:]
    ALT_3=ALT_3[0:21801,:]
    TB_4=TB_4[0:21801,:]
    DB_5=DB_5[0:21801,:]

    t_idn_star = t_idn
    yt_idn_star = yt_idn


    ### Training Data ###

    # For identification
    N_train = 21801

    idx = np.random.choice(t_idn_star.shape[0], N_train, replace=False)

    t_train = t_idn_star[idx, :]
    yt_train  = yt_idn[idx, :]
    ALB_train = ALB[idx, :]
    Cr_1_train = Cr_1[idx, :]
    OSM_2_train = OSM_2[idx, :]
    ALT_3_train = ALT_3[idx, :]
    TB_4_train = TB_4[idx, :]
    DB_5_train = DB_5[idx, :]


    if args.step1 :

        print('hello1! we now begin step1!111111111')

        # Train Neural net U(x,t)
        model.idn_u_train(N_iter=100000, t_=t_train, yt_=yt_train, ALB_=ALB_train, Cr_1_=Cr_1_train, OSM_2_=OSM_2_train ,ALT_3_=ALT_3_train, TB_4_=TB_4_train, DB_5_=DB_5_train)

        u_pred_identifier, f_pred_identifier = model.idn_predict(t_idn_star, yt_idn_star)

        #plotting for step1
        plotfigure(t_idn, ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier, step=1)

        saver = tf.train.Saver()
        save_path = saver.save(model.sess, checkpoint_file_step1)

        #Index output describing fitting degree
        Error_u(ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier)

        saver = tf.train.Saver()
        save_path = saver.save(model.sess, checkpoint_file_step1)
        print('goodbye1! we have finished step1!111111111')


    elif args.step2:

        saver = tf.train.Saver()
        save_path = saver.restore(model.sess, checkpoint_file_step1)

        #step2 begins
        print('hello2! we now begin step2!222222222')
        model.idn_f_train(N_iter=100000, t_=t_train, yt_=yt_train)

        saver = tf.train.Saver()
        save_path = saver.save(model.sess, checkpoint_file_step2)
        
        print('goodbye2! we have finished step2!222222222')


    elif args.step3 :

        print('hello3! we now begin step3!333333333')

        saver = tf.train.Saver()
        save_path = saver.restore(model.sess, checkpoint_file_step2)

        #step3 
        model.sol_train(N_iter=100000, t_=t_train, yt_=yt_train, ALB_=ALB_train, Cr_1_=Cr_1_train, OSM_2_=OSM_2_train ,ALT_3_=ALT_3_train, TB_4_=TB_4_train, DB_5_=DB_5_train)

        u_pred_identifier, f_pred_identifier = model.sol_predict(t_idn_star, yt_idn_star)

        #plotting for step3
        plotfigure(t_idn, ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier, step=3)


        saver = tf.train.Saver()
        save_path = saver.save(model.sess, checkpoint_file_step3)
        
        #Index output describing fitting degree
        Error_u(ALB, Cr_1, OSM_2, ALT_3, TB_4, DB_5, u_pred_identifier)

        saver = tf.train.Saver()
        save_path = saver.save(model.sess, checkpoint_file_step3)    
        print('goodbye3! we have finished step3!333333333')


    elif args.step4 :

        print('hello4! we now begin step4!444444444')

        saver = tf.train.Saver()
        save_path = saver.restore(model.sess, checkpoint_file_step3)
        ### Load Data ###

        data_idn = scipy.io.loadmat('./Data/data_cd_hptal.mat')
        t_idn = data_idn['tt'].flatten()[:, None]
        yt_idn = data_idn['yy'].flatten()[:, None]
        ALB = data_idn['ALB'].flatten()[:, None]
        Cr_1 = data_idn['Cr'].flatten()[:, None]
        OSM_2 = data_idn['OSM'].flatten()[:, None]
        ALT_3 = data_idn['ALT'].flatten()[:, None]
        TB_4 = data_idn['TB'].flatten()[:, None]
        DB_5 = data_idn['DB'].flatten()[:, None]


        begin_time_point = 21801
        # m = args.prediction_time

        for m in [1000]:
        # for m in [100, 1000, 2400]:
            x_0 = ALB[begin_time_point][:, None]
            x_1 = Cr_1[begin_time_point][:, None]
            x_2 = OSM_2[begin_time_point][:, None]
            x_3 = ALT_3[begin_time_point][:, None]
            x_4 = TB_4[begin_time_point][:, None]
            x_5 = DB_5[begin_time_point][:, None]

            all_together = np.concatenate((ALB[begin_time_point], Cr_1[begin_time_point], OSM_2[begin_time_point], ALT_3[begin_time_point], TB_4[begin_time_point], DB_5[begin_time_point]),axis=0)
            all_together = np.array([all_together])

            x_old = np.concatenate((ALB[begin_time_point], Cr_1[begin_time_point], OSM_2[begin_time_point], ALT_3[begin_time_point], TB_4[begin_time_point], DB_5[begin_time_point]),axis=0)

            for i in range(m):
                F_pred = model.prediction_step4(x_0, x_1, x_2, x_3, x_4, x_5)
                yt_arrow = np.concatenate((yt_idn[begin_time_point+i],[0],[0],[0],[0],[0]),axis=0)
                x_old = np.concatenate((ALB[begin_time_point+i], Cr_1[begin_time_point+i], OSM_2[begin_time_point+i], ALT_3[begin_time_point+i], TB_4[begin_time_point+i], DB_5[begin_time_point+i]),axis=0)
                x_new = 0.01*(F_pred[0]+yt_arrow*0.12)+x_old
                # x_new = 0.01*(F_pred[0]+yt_arrow)+x_old
                x_new_matrix = np.array([x_new])
                all_together = np.concatenate((all_together,x_new_matrix),axis=0)
                x_0 = x_new[:,None][0][:,None]
                x_1 = x_new[:,None][1][:,None]
                x_2 = x_new[:,None][2][:,None]
                x_3 = x_new[:,None][3][:,None]
                x_4 = x_new[:,None][4][:,None]
                x_5 = x_new[:,None][5][:,None]
                

            y_1 = ALB[begin_time_point:begin_time_point+m+1]
            y_2 = Cr_1[begin_time_point:begin_time_point+m+1]
            y_3 = OSM_2[begin_time_point:begin_time_point+m+1]
            y_4 = ALT_3[begin_time_point:begin_time_point+m+1]
            y_5 = TB_4[begin_time_point:begin_time_point+m+1]
            y_6 = DB_5[begin_time_point:begin_time_point+m+1]
            x = t_idn[begin_time_point:begin_time_point+m+1]

            j = 1
            for y in [y_1]:
            # for y in [y_1, y_2, y_3, y_4, y_5, y_6]:
                plt.figure(j)
                plt.plot(x, y,'--b',label='true value of ALB')  
                plt.plot(x, all_together[:,j-1],'--r',label='prediction with injection of ALB')  
                plt.legend(['true value of ALB','prediction with injection of ALB','prediction without injection of ALB'])
                # plt.legend(['true value of ALB','prediction with injection of ALB','prediction without injection of ALB'],fontsize='large')
                plt.savefig(f'./alpha%d/pred_%d_%d_1.eps'%(args.alpha, m, j))
                j = j+1


        print('goodbye4! we have finished step4!444444444')

    else:
        pass



