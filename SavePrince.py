
import numpy as np



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
    def __init__(self):
        ## frame per time (sec)
        self.FPT = 20

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
        gameDisplay.blit(textSurf, textRect)

    def _text_objects(self,text, color, font):
        font = pygame.font.SysFont(None, font)
        textSurface = font.render(text, True, color)
        return textSurface, textSurface.get_rect()

    def intro(self,npix_x,npix_y):
        Intro = True
        while Intro:
            gameDisplay.fill(black)
            self.message_to_screen("Save your prince", green, 30,
                              x=npix_x / 2, y=npix_y * 1 / 4)
            self.message_to_screen("before the snake monster gets you!", green, 30,
                              x=npix_x / 2, y=npix_y * 2 / 4)
            self.message_to_screen("Press any key to get started.", green, 30,
                              x=npix_x / 2, y=npix_y * 3 / 4)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    Intro = False

    def report_status(self,myscore, start, font=30):
        smallfont = pygame.font.SysFont("comicsansms", font)

        now = time.time()
        strtime = self._get_time_diff(start, now)
        text = smallfont.render("My prince: " + str(myscore) + strtime,
                                True, white)
        gameDisplay.blit(text, [15, 15]) ## topleft corner

    def _get_time_diff(self,start, now):
        diff = now - start
        min, sec = int(diff // 60), int(diff % 60)
        timestr = ' Time {:02.0f}:{:02.0f}'.format(min, sec)

        return (timestr)


    def display_screen(self,Nprinces, start,
                       state,
                       goal_xy,
                       monster_size):
        princess_xy = state["princess"]
        monster_x = state["monster"][0]
        monster_y= state["monster"][1]

        gameDisplay.fill(black)
        gameDisplay.blit(img,princess_xy)
        gameDisplay.blit(img_prince, goal_xy)


        for lx,ly in zip(monster_x,monster_y):
            pygame.draw.rect(gameDisplay,purple,
                             (lx,ly,
                              monster_size,
                              monster_size))

        self.report_status(Nprinces, start, font=30)
        pygame.display.update()
        clock.tick(self.FPT)  ## 20 frames / sec


if __name__ == '__main__':
    import pygame
    from color import *
    import time
    pygame.init()

    ## define environment/game
    env = SavePrince()

    img = pygame.image.load('./pic/princess.png')
    img_prince = pygame.image.load('./pic/prince.png')
    img_prince_happy = pygame.image.load('./pic/prince_happy.png')
    pygame.display.set_caption("princess's game")
    pygame.display.set_icon(img)

    wait_sec_after_get_prince = 1
    Nprinces = 0
    gameDisplay = pygame.display.set_mode((env.npix_x, env.npix_y))
    start = time.time()
    clock = pygame.time.Clock()
    interface = GameInterface()
    interface.intro(env.npix_x, env.npix_y)


    while not env.game_over:
        action = interface.check_keyboard_actions(pygame)
        reward, game_over = env.act(action)

        interface.display_screen(Nprinces,
                                 start,
                                 env.state,
                                 env.state_static,
                                 env.monster_size)

        #image_array = pygame.surfarray.array3d(gameDisplay)

        if reward == 1:
            interface.message_to_screen("FOUND HIM <3", magenta, 80,
                                        env.npix_x/2,env.npix_y/2)
            gameDisplay.blit(img_prince_happy, env.state_static)
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
