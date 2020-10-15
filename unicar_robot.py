from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.UnicarRobotArm import UnicarRobotArm
from pyrep.robots.end_effectors.unicarrobot_suction_cup import UnicarRobotGreifer
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import JointType,JointMode
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

POS_MIN,POS_MAX = [-0.2,1.35,1.75],[0.2,1.75,1.75]

class GraspEnv(object):
    def __init__(self,headless, control_mode ='joint_velocity'):
        self.headless = headless
        self.reward_offset = 10.0
        self.reward_range = self.reward_offset
        self.penalty_offset = 1
        #self.penalty_offset = 1.
        self.fall_down_offset = 0.1
        self.metadata = [] #gym env argument
        self.control_mode = control_mode

        #launch and setup the scene, and set the proxy variables in present of the counterparts in the scene
        self.pr = PyRep()
        if control_mode == 'end_position':
            SCENE_FILE = join(dirname(abspath(__file__)),'./scenes/UnicarRobot_ik.ttt')
        elif control_mode == 'joint_velocity':
            SCENE_FILE = join(dirname(abspath(__file__)),'./scenes/UnicarRobot.ttt')
        self.pr.launch(SCENE_FILE,headless=headless)
        self.pr.start()
        self.agent = UnicarRobotArm()#drehkranz + UR10
        #self.gripper = UnicarRobotGreifer()#suction
        #self.suction = UnicarRobotGreifer()
        self.suction = Shape("UnicarRobotGreifer_body_sub0")
        self.proximity_sensor = ProximitySensor('UnicarRobotGreifer_sensor')
        self.table = Shape('UnicarRobotTable')
        self.carbody = Shape('UnicarRobotCarbody')
        
        if control_mode == 'end_position':
            self.agent.set_control_loop_enabled(True)
            self.action_space = np.zeros(4)
        elif control_mode == 'joint_velocity':
            self.agent.set_control_loop_enabled(False)
            self.action_space = np.zeros(7)
        else:
            raise NotImplementedError
        #self.observation_space = np.zeros(17)
        self.observation_space = np.zeros(20)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape("UnicarRobotTarget")#Box
        self.agent_ee_tip = self.agent.get_tip()
        self.tip_target = Dummy('UnicarRobotArm_target')
        self.tip_pos = self.agent_ee_tip.get_position()
        
        # set a proper initial robot gesture or tip position
        if control_mode == 'end_position':
            initial_pos = [0, 1.5, 1.6]
            self.tip_target.set_position(initial_pos)
            #one big step for rotatin is enough, with reset_dynamics = True, set the rotation instantaneously
            self.tip_target.set_orientation([0,0,0],reset_dynamics=True)
        elif control_mode == 'joint_velocity':
            self.initial_joint_positions = [0,0,0,0,0,0,0]
            self.agent.set_joint_positions(self.initial_joint_positions)
        self.pr.step()
        self.initial_tip_positions = self.agent_ee_tip.get_position()
        self.initial_target_positions = self.target.get_position()
    
    def _get_state(self):
        '''
        Return state containging arm joint positions/velocities & target position.
        '''
        return np.array(self.agent.get_joint_positions()+self.agent.get_joint_velocities()+self.agent_ee_tip.get_position()+self.agent_ee_tip.get_orientation())  #all 20

    def _is_holding(self):
        '''
        Return the state of holding the UnicarRobotTarget or not, return bool value
        '''
        if self.proximity_sensor.is_detected(self.target) == True:
            return True
        else:
            return False

    def _move(self, action, bounding_offset=0.15, step_factor=0.2, max_itr=20, max_error=0.05, rotation_norm=5):
        pos = self.suction.get_position()

        if pos[0]+action[0] > POS_MIN[0]-bounding_offset and pos[0]+action[0] < POS_MAX[0]-bounding_offset \
            and pos[1]+action[1] > POS_MIN[1]-bounding_offset and pos[1]+action[1] < POS_MAX[1]-bounding_offset \
            and pos[2]+action[2] > POS_MIN[2]-2*bounding_offset:

            ori_z = -self.agent_ee_tip.get_orientation()[2]# the minus is because the mismatch between the set_orientation() and get_orientation()
            #ori_z = self.agent_ee_tip.get_orientation()[2]
            target_pos = np.array(self.agent_ee_tip.get_position()+np.array(action[:3]))
            diff = 1
            itr = 0
            while np.sum(np.abs(diff)) > max_error and itr < max_itr:
                itr += 1
                # set pos in small step
                cur_pos = self.agent_ee_tip.get_position()
                diff = target_pos - cur_pos
                pos = cur_pos+step_factor*diff
                self.tip_target.set_position(pos.tolist())
                self.pr.step()

            ori_z += rotation_norm*action[3]
            self.tip_target.set_orientation([0,np.pi,ori_z])
            self.pr.step()

        else:
            print("Potantial Movement Out of the Bounding Box!")
            self.pr.step()

    def reinit(self):
        self.shutdown()
        self.__init__(self.headless)
    
    def reset(self,random_target=False):
        '''
        Get a random position within a cuboid and set the target position
        '''
        # set target object
        if random_target:
            pos = list(np.random.uniform(POS_MIN,POS_MAX))
            self.target.set_position(pos)
        else:
            self.target.set_position(self.initial_target_positions)
        self.target.set_orientation([0,0,0])
        self.pr.step()

        #set end position to be initialized
        if self.control_mode == 'end_position':
            self.agent.set_control_loop_enabled(True)# IK mode
            self.tip_target.set_position(self.initial_tip_positions)
            self.pr.step()
            itr = 0
            max_itr = 10
            while np.sum(np.abs(np.array(self.agent_ee_tip.get_position()-np.array(self.initial_tip_positions))))>0.1 and itr<max_itr:
                itr+=1
                self.step(np.random.uniform(-0.2,0.2,4))
                self.pr.step()
        elif self.control_mode == 'joint_velocity':#JointMode Force
            self.agent.set_joint_positions(self.initial_joint_positions)
            self.pr.step()
        
        #set collidable, for collision detection
        #self.gripper.set_collidable(True)
        self.suction.set_collidable(True)
        self.target.set_collidable(True)

        return self._get_state() #return the current state of the environment 

    def step(self,action):
        '''
        move the robot arm according to the control mode
        if control_mode == 'end_position' then action is 3 dim of tip (end of robot arm) position values + 1 dim rotation of suction
        if control_mode == 'joint_velocity' then action is 7 dim of joint velocity + 1 dim of rotation of suction
        '''
        #initialization
        done=False#episode finishes
        reward=0
        hold_flag=False#hold the object or not
        if self.control_mode == 'end_position':
            if action is None or action.shape[0] != 4: #check action is valid
                print('No actions or wrong action dimensions!')
                action = list(np.random.uniform(-0.1,0.1,4))
            self._move(action)
        elif self.control_mode == 'joint_velocity':
            if action is None or action.shape[0] != 7:#???
                print('No actions or wrong action dimensions!')
                action = list(np.random.uniform(-1, 1, 7))
            self.agent.set_joint_target_velocities(action)
            self.pr.step()
        
        else:
            raise NotImplementedError

        #ax,ay,az = self.gripper.get_position()#gripper position
        ax,ay,az = self.suction.get_position()#suction position
        if math.isnan(ax):#capture the broken suction cases during exploration
            print("Suction position is nan.")
            self.reinit()
            done=True

        desired_position_tip = [0.0,1.5513,1.74]
        desired_orientation_tip = [-np.pi,0,0.001567]
        tip_x,tip_y,tip_z = self.agent_ee_tip.get_position()#end_effector position
        tip_row,tip_pitch,tip_yaw = self.agent_ee_tip.get_orientation()#end_effector orientation
        tx,ty,tz = self.target.get_position()#box position
        offset = 0.312 #augmented reward: offset of target position above the target object 
        #square distance between the gripper and the target object
        sqr_distance = np.sqrt((tip_x-desired_position_tip[0])**2 + (tip_y-desired_position_tip[1])**2 + (tip_z-desired_position_tip[2])**2)
        sqr_orientation=np.sqrt((tip_row-desired_orientation_tip[0])**2 + (tip_pitch-desired_orientation_tip[1])**2+(tip_yaw-desired_orientation_tip[2])**2)


        ''' for visual-based control only, large time consumption! '''
        # current_vision = self.vision_sensor.capture_rgb()  # capture a screenshot of the view with vision sensor
        # plt.imshow(current_vision)
        # plt.savefig('./img/vision.png')
        desired_orientation = [0,0,-np.pi/2]
        desired_orientation_tip = [-np.pi,0,0.001567]
        #Enable the suction if close enough to the object and the object is detected with the proximity sensor
        if sqr_distance<0.001 and self.proximity_sensor.is_detected(self.target)==True and sqr_orientation<0.001:
            #make sure the suction is not worked
            self.suction.release()
            self.pr.step()
            self.suction.grasp(self.target)
            self.pr.step()

            if self._is_holding():
                reward+=self.reward_offset
                done=True
                hold_flag = True
            else:
                self.suction.release()
                self.pr.step()
        else:
            pass

        #the base reward is negative distance from suction to target 
        #reward -= (np.sqrt(sqr_distance))
        
        #case when the object is fall off the table
        if tz < self.initial_target_positions[2]-self.fall_down_offset:#tz is target(box) position in z direction
            done = True
            reward -= self.reward_offset
    
        # Augmented reward for orientation: better grasping gesture if the suction has vertical orientation to the target object
        
        desired_position_tip = [0.0,1.5513,1.74]
        tip_x,tip_y,tip_z = self.agent_ee_tip.get_position()
        tip_row,tip_pitch,tip_yaw = self.agent_ee_tip.get_orientation()
        
        reward -= (np.sqrt((tip_x-desired_position_tip[0])**2 + (tip_y-desired_position_tip[1])**2 + (tip_z-desired_position_tip[2])**2) + np.sqrt((tip_row-desired_orientation_tip[0])**2 + (tip_pitch-desired_orientation_tip[1])**2+(tip_yaw-desired_orientation_tip[2])**2))
        
        #Penalty for collision with the table
        if self.suction.check_collision(self.table) or self.suction.check_collision(self.carbody) or self.agent.check_collision(self.table) or self.suction.check_collision(self.target) or self.agent.check_collision(self.target):
            reward -= self.penalty_offset

        if math.isnan(reward):
            reward=0.

        return self._get_state(),reward,done,{'finished':hold_flag}

    def shutdown(self):
        '''close the simulator'''
        self.pr.stop()
        self.pr.shutdown()

if __name__ == '__main__':
    CONTROL_MODE='joint_velocity'  # 'end_position' or 'joint_velocity'
    env=GraspEnv(headless=False, control_mode=CONTROL_MODE)
    for eps in range(30):
        env.reset()
        for step in range(30):
            if CONTROL_MODE=='end_position':
                action=np.random.uniform(-0.2,0.2,4)  #  4 dim control for 'end_position': 3 positions and 1 rotation (z-axis)
            elif CONTROL_MODE=='joint_velocity':
                action=np.random.uniform(-3.14,3.14,7)
            else:
                raise NotImplementedError
            try:
                env.step(action)
            except KeyboardInterrupt:
                print('Shut Down!')
    env.shutdown()


