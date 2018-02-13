# To run the game, on terminal
#
# python2 SavePrince.py
import numpy as np
import pygame
from color import *
import time

class SavePrince(object):
    def __init__(self,
                 speed_monster=1,
                 monster_size=20,
                 goal_size=50,
                 princess_size=20,
                 speed = 10,
                 len_monster = 80,
                 ## N of pixcel along y axis
                 npix_x=400,
                 ## N of pixcel along x axis
                 npix_y=400):
        self.monster_size = monster_size
        self.goal_size = goal_size
        self.princess_size = princess_size
        ## N of pixcel along y axis
        self.npix_y = npix_y
        ## N of pixcel along x axis
        self.npix_x = npix_x
        ## the length of sneak like monster
        self.len_monster = len_monster
        ## speed of princess movement every frame
        self.speed = speed # 10
        ## speed of monster movement
        self.speed_monster = speed_monster
        self.Nprincess = 0

        self.reset()

    def reset(self):
        self.reset_static_state()
        self.reset_state()
        self.reset_speed()
        self.game_over = 0
        self.find_prince = False

    def reset_state(self):
        '''
        :return:
         self.state contains: locations of princess and monsters:

         self.state["princess"] = (x-cord of princess, y-cord of princess)
         self.state["monster"] = ((x-cord of monster's tail),
                                  (y-cord of monster's tail))

        monster has a length <slen_monster>
        '''

        (goal_x, goal_y) = self.state_static

        lead_x = self.npix_x / 2.  ## "leader" of the block
        lead_y = self.npix_y / 2.  ##
        eps = self.goal_size / 2.

        monster_x = np.random.choice([goal_x + eps, goal_x - eps])
        monster_y = np.random.choice([goal_y + eps, goal_y - eps])


        monster_xs = [monster_x]*self.len_monster
        monster_ys = [monster_y]*self.len_monster

        self.state = {}
        self.state["princess"] = [lead_x,lead_y]
        self.state["monster"] =  [monster_xs, monster_ys]
    def reset_static_state(self):
        '''
        :return:
         state_static contains the location of goal coordinates
         the goal coordinates do not change during the single play
        '''
        if (self.goal_size  >= self.npix_x) or (self.goal_size  >= self.npix_y):
            print("goal_size {:} must be less than the npix_x {:} and npix_y {:}".format(
                self.goal_size,self.npix_x,self.npix_y))
            raise Exception

        goal_x = np.random.choice(range(self.goal_size / 2,
                                        self.npix_x - self.goal_size / 2))
        goal_y = np.random.choice(range(self.goal_size / 2,
                                        self.npix_y - self.goal_size / 2))

        self.state_static = (goal_x, goal_y)



    def reset_speed(self):
        '''
        :return:
         define the speed of movement for princess and monster
        '''
        self.lead_x_change, self.lead_y_change = 0, 0
        self.monster_x_change, self.monster_y_change = 0, 0




    ########


    def _update_state(self,action):
        self.update_princess(action)
        self.update_monster()


    def update_monster(self):
        self._update_monster_direction()
        # monster's head is saved at the 0th position
        head_x, head_y = self._update_coordinates(
                        self.state["monster"][0][0],
                        self.state["monster"][1][0],
                        self.monster_x_change,
                        self.monster_y_change)
        self.state["monster"][0].insert(0, head_x)
        self.state["monster"][1].insert(0, head_y)
        if len(self.state["monster"][0]) > self.len_monster:
            for i in range(2):
                self.state["monster"][i] = self.state["monster"][i][:self.len_monster]

    def update_princess(self,action):
        '''
        :param action:
            # action is a scalar
            # 0 = "L", 1 = "R", 2 = "D", 3 = "U"
        :return:
            update the princess info.
        '''

        if action == 0: #L
            self.lead_x_change -= self.speed
            self.lead_y_change = 0
        elif action == 1: #R
            self.lead_x_change += self.speed
            self.lead_y_change = 0
        elif action == 2: #D
            self.lead_y_change += self.speed
            self.lead_x_change = 0
        elif action == 3: #U
            self.lead_y_change -= self.speed
            self.lead_x_change = 0
        elif action == 4: ## Do nothing
            self.lead_x_change = 0
            self.lead_y_change = 0

        self.state["princess"][0], self.state["princess"][1] = self._update_coordinates(
        self.state["princess"][0], self.state["princess"][1],
        self.lead_x_change, self.lead_y_change)

    def _update_coordinates(self,
                            lead_x, lead_y,
                            lead_x_change,
                            lead_y_change):
        lead_x += lead_x_change
        lead_x = self._within_screen(lead_x, self.npix_x)
        lead_y += lead_y_change
        lead_y = self._within_screen(lead_y, self.npix_y)
        return (lead_x, lead_y)


    def _within_screen(self,load_x, npix_x):
        '''
        :param load_x: x-coordinate scalar
        :param npic_x: the maximum number of pixcel along this axis
        :return:
        if lead_x is outside of [0,npix_x] then lead_x are truncted
        '''
        if load_x < 0:
            load_x = 0
        elif load_x > (npix_x - self.princess_size):
            load_x = npix_x - self.princess_size
        return (load_x)




    def _update_monster_direction(self):
        '''
        :return:
        update
        self.monster_x_change and self.monster_y_change
        '''
        lead_x, lead_y = self.state["princess"]
        monster_x, monster_y = self.state["monster"][0][0],\
                               self.state["monster"][1][0]
        diffx = lead_x - monster_x
        diffy = lead_y - monster_y

        xlab = "R" if diffx > 0 else "L"
        ylab = "U" if diffy > 0 else "D"

        p1, p2 = np.abs(diffx), np.abs(diffy)
        tot = float(p1 + p2)
        if tot == 0:
            p = [0.5,0.5]
        else:
            p = [p1 / tot, p2 / tot]
        rnd = np.random.choice([xlab,ylab],p=p)

        if rnd == "L":
            self.monster_x_change -= self.speed_monster
            self.monster_y_change = 0
        elif rnd == "R":
            self.monster_x_change += self.speed_monster
            self.monster_y_change = 0
        elif rnd == "D":
            self.monster_y_change -= self.speed_monster
            self.monster_x_change = 0
        elif rnd == "U":
            self.monster_y_change += self.speed_monster
            self.monster_x_change = 0

    def _get_reward(self):
        if self.check_monster_hit_princess():
            self.game_over = 1
            return -1
        elif self.check_princess_find_prince():
            return 1
        else:
            return 0

    def check_monster_hit_princess(self,size=None):
        '''
        :return:
         True if monster hits princess
        '''
        if size is None:
            size = self.monster_size
        cent_x, cent_y = self.state["princess"] ## record the left top coner of princess
        xvec = np.array(self.state["monster"][0]) ## record the left top coner
        yvec = np.array(self.state["monster"][1]) ## of every pixcel

        ## center of (x,y) princess cordinate
        cent_x += self.princess_size/2
        cent_y += self.princess_size/2

        TFx_min =  xvec <= cent_x
        TFx_max = cent_x <= (xvec + size)

        TFy_min = yvec <= cent_y
        TFy_max = cent_y <= (yvec + size)
        TF = np.any(TFx_min & TFx_max & TFy_min & TFy_max)
        return(TF)


    def check_princess_find_prince(self,size=None):
        if size is None:
            size = self.goal_size
        (goal_x, goal_y) =  self.state_static
        (cent_x, cent_y) = self.state["princess"]

        ## center of (x,y) princess cordinate
        cent_x += self.princess_size/2
        cent_y += self.princess_size/2

        TFx_min = goal_x < cent_x
        TFx_max = cent_x < goal_x + size
        TFy_min = goal_y < cent_y
        TFy_max = cent_y < goal_y + size
        TF = TFx_min and TFx_max and TFy_min and TFy_max
        return (TF)



    def _is_over(self):
        return self.game_over

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return reward, game_over


class GameInterface(object):

    def __init__(self,pygame,npix_x, npix_y, FPT,env):
        ## frame per time (sec)
        self.FPT = FPT
        self.wait_sec_after_get_prince = 1
        self.npix_x, self.npix_y = npix_x, npix_y
        self.pygame = pygame
        self.pygame.display.set_caption("princess's game")
        self.define_plot_image(env)
        self.gameDisplay = self.pygame.display.set_mode(
            (self.npix_x, self.npix_y),self.pygame.RESIZABLE)
        self.setup()

    def setup(self):

        self.start = time.time()
        self.clock = self.pygame.time.Clock()

    def action_to_string(self,action):
        actions = ["Left", "Right", "Down", "Up", "Stop"]

        if action is None:
            return("   ")
        elif isinstance(action,list):
            action = action[0]

        return(actions[action])

    def define_plot_image(self,env):
        '''
        :param drawing:
        :return:
        '''

        self.img = pygame.image.load('./pic/princess.png')
        self.img_prince = pygame.image.load('./pic/prince.png')
        self.img_prince_happy = pygame.image.load('./pic/prince_happy.png')
        self.pygame.display.set_icon(self.img)
        self.princess_size = env.princess_size
        self.prince_size = env.goal_size
        self.plt_princess = lambda xy: self.gameDisplay.blit(self.img, xy)
        self.plt_prince   = lambda xy: self.gameDisplay.blit(self.img_prince, xy)
        ## goal_xy is top left coner
        self.plt_prince_happy = lambda xy: self.gameDisplay.blit(self.img_prince_happy, xy)

    def check_keyboard_actions(self,pygame):
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return 0
                elif event.key == pygame.K_RIGHT:
                    return 1
                elif event.key == pygame.K_DOWN:
                    return 2
                elif event.key == pygame.K_UP:
                    return 3
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or\
                            event.key == pygame.K_RIGHT or\
                            event.key == pygame.K_DOWN or\
                            event.key == pygame.K_UP:
                    return 4 ## Do nothing

    def message_to_screen(self,msg, color, font,
                          x=None,y=None):

        textSurf, textRect = self._text_objects(msg, color, font)
        textRect.center = (x, y)
        self.gameDisplay.blit(textSurf, textRect)

    def _text_objects(self,text, color, font):
        font = pygame.font.SysFont(None, font)
        textSurface = font.render(text, True, color)
        return textSurface, textSurface.get_rect()

    def intro(self):
        pygame = self.pygame
        Intro = True
        while Intro:
            font_size = int(self.npix_x*0.08)
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

    def report_status(self,myscore, action, reward, font=10):
        font = self.npix_x/15
        if font >= 1:
            smallfont = self.pygame.font.SysFont("comicsansms", font)

            now = time.time()
            strtime = self._get_time_diff(self.start, now)
            mystring = "Prince:" + str(myscore) + strtime + \
                       " Reward:{:4.3f}".format(reward) +\
                       " Action:{:5}".format(self.action_to_string(action))
            text = smallfont.render(mystring,True, white)

            loc=30.0/(self.npix_x)
            self.gameDisplay.blit(text, [loc, loc]) ## topleft corner


    def _get_time_diff(self,start, now):
        diff = now - start
        min, sec = int(diff // 60), int(diff % 60)
        timestr = ' Time {:02.0f}:{:02.0f}'.format(min, sec)

        return (timestr)

    def display_screen(self,
                       state,
                       Nprincess,
                       goal_xy,
                       monster_size,
                       action,reward):
        princess_xy = state["princess"]
        monster_x = state["monster"][0]
        monster_y= state["monster"][1]

        self.gameDisplay.fill(black)
        self.plt_prince(goal_xy)
        self.plt_princess(princess_xy)
        #gameDisplay.blit(self.img,princess_xy) ## princess_xy is top left coner
        #gameDisplay.blit(self.img_prince, goal_xy)  ## goal_xy is top left coner

        for lx,ly in zip(monster_x,monster_y):
            self.pygame.draw.rect(self.gameDisplay,purple,
                             (lx,ly, ## top left corner
                              monster_size, ## width
                              monster_size)) ## height


        self.report_status(Nprincess, action, reward)
        self.pygame.display.update()

        self.clock.tick(self.FPT)  ## 20 frames / sec



if __name__ == '__main__':
    pygame.init()

    ## define environment/game
    env = SavePrince()


    pygame.display.set_caption("princess's game")


    wait_sec_after_get_prince = 1
    Nprinces = 0
    start = time.time()
    clock = pygame.time.Clock()
    interface = GameInterface(pygame,
                                      env.npix_x,
                                      env.npix_y,
                                      env=env,
                                      FPT=20)
    interface.intro()
    clock = interface.setup()

    while not env.game_over:
        action = interface.check_keyboard_actions(pygame)
        reward, game_over = env.act(action)

        interface.display_screen(env.state,
                                 Nprinces,
                                 env.state_static,
                                 env.monster_size,
                                 action, reward)

        #image_array = pygame.surfarray.array3d(gameDisplay)

        if reward == 1:
            interface.message_to_screen("FOUND HIM <3", magenta, 80,
                                        env.npix_x/2,env.npix_y/2)
            interface.plt_prince_happy(env.state_static)
            pygame.display.update()
            pygame.time.wait(wait_sec_after_get_prince * 1000)
            Nprinces += 1
            env.reset()



    interface.message_to_screen("DEAD", red, 100,env.npix_x/2,env.npix_y/2)
    interface.message_to_screen("Press Q to quit", white, 19,
                                 env.npix_x/2, env.npix_y * 2/3)
    pygame.display.update()

    gameFinalExit = False
    while not gameFinalExit:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    print("currently note supported")
                    gameFinalExit = True
                elif event.key == pygame.K_q:
                    gameFinalExit = True
    pygame.quit()
    quit()
