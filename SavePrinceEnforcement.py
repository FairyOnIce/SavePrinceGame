## Reference
## https://gist.github.com/EderSantana/c7222daa328f0e885093
import numpy as np
import pygame
from color import *
import time
import json
import matplotlib.pyplot as plt
from SavePrince import SavePrince, GameInterface


class SavePrinceReEnforce(SavePrince):
    def _get_reward(self):
        '''
        keep
        :return:
        '''
        tot = self.npix_x * np.sqrt(2)
        if self.check_monster_hit_princess(self.monster_size):
            self.game_over = 1
            reward = -2
        else:
            ## close to monster
            #(x, y, xm, ym) = self.find_closest_monster()
            #reward = (np.sqrt((x-xm)**2 + (y-ym)**2))/tot - 1/2.0
            reward = 0
        if self.check_princess_find_prince(self.goal_size):
            self.Nprincess += 1
            self.find_prince = True
            reward += 2

        else:
            x,y = self.state["princess"]
            xm,ym = self.state_static
            reward += 1 - (np.sqrt((x - xm) ** 2 + (y - ym) ** 2)) / tot
            reward -= 1/2.0
        return(reward)

    def find_closest_monster(self):
        '''
        keep
        :return:
         (xm,ym) : the x,y coordinates of monster closest to the princess
        '''
        xvec = np.array(self.state["monster"][0])
        yvec = np.array(self.state["monster"][1])
        x, y = self.state["princess"]
        idx = np.argmin((xvec - x)**2 + (yvec - y)**2)
        xm, ym = xvec[idx], yvec[idx]
        return(x, y, xm, ym)


    def game_loop(self,interface,epoch):
        '''
        keep
        :param interface:
        :param epoch:
        :return:
        '''
        Naction = 0

        loss = 0.
        self.reset()
        # get initial input
        input_tm1 = rl.frame2array(interface.gameDisplay)
        while not env.game_over:
            Naction += 1

            action = rl.predict_with_randomness(input_tm1)
            #action = interface.check_keyboard_actions()

            reward, game_over = env.act(action)
            interface.display_screen(self.state,
                                     self.Nprincess,
                                     self.state_static,
                                     self.monster_size,
                                     action, reward)

            input_t = rl.frame2array(interface.gameDisplay)
            rl.remember([input_tm1,  ## (1, npix_x, npix_y, dim)
                           action,     ## scalar
                           reward,     ## scalar
                           input_t],   ## (1, npix_x, npix_y, dim)
                           game_over)
            inputs, targets = rl.get_batch(batch_size=batch_size)
            loss += rl.model.train_on_batch(inputs, targets)
            input_tm1 = input_t

            print("Epoch {:03d}/{:03d} | Naction {:3}| input size {}| Loss {:.4f} | reward {:4.2f} | action {:}".format(
                    epoch,epochs,Naction,inputs.shape[0],loss/Naction, reward, action))

            if epoch % save_freq == 0:
                ## record every 10 epoch
                plt.imshow(input_tm1.reshape((env.npix_x,env.npix_y)),
                           interpolation='none', cmap='gray')
                plt.savefig(dir_result + "Epoch{:03d}_action{:03d}.png".format(epoch,Naction))

        return(loss,Naction)

class GameInterfaceRenforce(GameInterface):


    def define_plot_image(self,env):
        '''
        :param drawing:
        :param env: SavePrince object require donly if drawing=False
        :return:
        '''

        self.princess_size = env.princess_size
        self.prince_size = env.goal_size
        self.plt_princess = lambda xy: pygame.draw.rect(self.gameDisplay, princess_skin,
                          (xy[0], xy[1], self.princess_size, self.princess_size))
        self.plt_prince   = lambda xy: pygame.draw.rect(self.gameDisplay, pcince_skin,
                          (xy[0], xy[1], self.prince_size,   self.prince_size))
        self.plt_prince_happy = lambda xy: pygame.draw.rect(self.gameDisplay, pcince_skin,
                          (xy[0], xy[1], self.prince_size,   self.prince_size))




    def intro(self):
        pygame = self.pygame
        Intro = True
        while Intro:
            font_size = int(self.npix_x*0.1)
            self.gameDisplay.fill(black)
            self.message_to_screen("Save your prince", green, font_size,
                              x=self.npix_x / 2, y=self.npix_y * 1 / 4)
            self.message_to_screen("before the snake monster gets you!", green, font_size,
                              x=self.npix_x / 2, y=self.npix_y * 2 / 4)
            self.message_to_screen("Press any key to get started.", green, font_size,
                              x=self.npix_x / 2, y=self.npix_y * 3 / 4)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    Intro = False












class ReinforcementLearning(object):

    def __init__(self,npix_x,npix_y, dim = 3, max_memory=100, discount=.9):
        self.epsilon = .1  # exploration
        self.num_actions = 5  # ["L","R","U","D","Do nothing"]
        self.hidden_size = 100
        self.batch_size = 50
        self.npix_x, self.npix_y = npix_x, npix_y
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.define_model(dim)

    def define_model(self,dim=3):
        from keras.models import Sequential
        from keras import layers
        self.dim = dim
        model = Sequential()
        model.add(layers.Flatten(input_shape=(self.npix_x,
                                              self.npix_y,
                                              self.dim)))
        model.add(layers.Dense(self.hidden_size,
                        activation='relu'))
        model.add(layers.Dense(self.hidden_size, activation='relu'))
        model.add(layers.Dense(self.num_actions))
        model.compile("adam", "mse")

        self.model = model



    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self,batch_size=10):
        '''
        :param model:  keras model object
        :param batch_size:

        :return:

        self.memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over]

        state_t.shape = (npix_x,npix_y,3)
        '''
        len_memory = len(self.memory)
        num_actions = self.model.output_shape[-1]
        npix_x, npix_y = self.memory[0][0][0].shape[1:3]
        inputs = np.zeros((min(len_memory, batch_size),
                           npix_x, npix_y,self.dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t ## (1, 3, 3, 1)
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = self.model.predict(state_t)[0] # 5 by 1
            Q_sa = np.max(self.model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets

    def frame2array(self,gameDisplay):
        input_t = pygame.surfarray.array3d(gameDisplay)
        if self.dim < 3:
            input_t = self.converter(input_t)
        shape = [1] + list(input_t.shape)
        return(input_t.reshape(shape))

    def converter(self,x):
        # https://stackoverflow.com/questions/46836358/keras-rgb-to-grayscale
        # x has shape (batch, width, height, channels)
        return (0.21 * x[ :, :, :1]) + (0.72 * x[ :, :, 1:2]) + (0.07 * x[ :, :, -1:])
    def predict_with_randomness(self,input_tm1):

        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions, size=1)
        else:
            q = self.model.predict(input_tm1)
            action = np.argmax(q[0])
        action = np.asscalar(action)
        return(action)

if __name__ == '__main__':

    dir_result = "./result/"
    try:
        os.makedirs(dir_result)
    except:
        pass
    dim = 1  ## 1 means gray scale and 3 means RGB
    batch_size = 50

    speed_monster = 2
    monster_size = 4
    goal_size = 3
    princess_size = 5#20
    speed = 5#10 ## speed for princess
    len_monster = 5 #80
    npix_x = 50#400
    npix_y = 50#400
    FPT = 2
    save_freq = 50
    epochs = 1000
    pygame.init()


    ## define environment/game
    env = SavePrinceReEnforce(speed_monster,
                     monster_size,
                     goal_size,
                     princess_size,
                     speed,
                     len_monster,
                     npix_x,
                     npix_y)


    interface = GameInterfaceRenforce(pygame,
                              env.npix_x,
                              env.npix_y,
                              env=env,
                              FPT=FPT)

    interface.intro()


    ## NEW

    rl = ReinforcementLearning(env.npix_x, env.npix_y,
                                   max_memory = batch_size*3,
                                   dim = dim,discount=0.9)

    #rl.model.load_weights(dir_result + "model_{:04fd}.h5".format(950))
    history = []
    for e in range(epochs):
        if e > 1000:e += 1000

        result = env.game_loop(interface,e)
        history.append(result)
        if e % save_freq == 0:
            np.savetxt(dir_result + "history.csv",history,delimiter=",")
            fnm = "model_{:04d}".format(e)
            rl.model.save_weights(dir_result + fnm + ".h5", overwrite=True)
            with open(dir_result + fnm + ".json", "w") as outfile:
                json.dump(rl.model.to_json(), outfile)

    pygame.quit()
    quit()
