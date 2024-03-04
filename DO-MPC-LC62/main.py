#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import pdb
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from casadi.tools import *

sys.path.append('../../')
import argparse
import time

import do_mpc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rcParams
#from model import template_model
from mpc import template_mpc
from sim import template_simulator

parser = argparse.ArgumentParser(description='MPC')
parser.add_argument('--ref', default="4.0,4.0,4.0,0.5" , type=str,
                    help='target pose')
parser.add_argument('--idx', default=0 , type=int,
                    help='index')
parser.add_argument('--episode', default=500 , type=int,
                    help='episode')
args = parser.parse_args()

file_idx=args.idx
episode=args.episode

# LC62 model dynamics
def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # States struct (optimization variables): roll, pitch, yaw
    z = model.set_variable(var_type='_x', var_name='Height', shape=(1,1))
    Velocity= model.set_variable(var_type='_x', var_name='Velocity', shape=(2,1))
    # Attitude    = model.set_variable(var_type='_x', var_name='Attitude', shape=(4,1)) # quaternion 
    # Rate  = model.set_variable(var_type='_x', var_name='Rate', shape=(3,1)) # first derivation of roll, pitch and yaw angle
    #DD_angle = model.set_variable(var_type='_x', var_name='DD_angle', shape=(3,1)) # second derivation of roll, pitch and yaw angle
    
    # Input struct (optimization variables):
    inp = model.set_variable(var_type='_u', var_name='inp', shape=(4,1)) # Fr, Fp, theta, q
   
    # uncertain parameters:
    J = model.set_variable('_p', 'J', shape=(3,3))
    # Ixx = model.set_variable('_p',  'Ixx')
    # Iyy = model.set_variable('_p',  'Iyy')
    # Izz = model.set_variable('_p',  'Izz')
    # g=model.set_variable('_p',  'g')
    # m=model.set_variable('_p',  'm')

    
    # set deriv
    Vx = Velocity[0]
    Vz = Velocity[1]
    Fr, Fp, theta, q = inp[0], inp[1], inp[2], inp[3]

    dz = -sin(theta) * Vx + cos(theta) * Vz
    dVx = (Fp + Fx) / m - g * sin(theta) - q * Vz
    dVz = (Fr + Fz) / m + g * cos(theta) + q * Vx
    
    # Differential equations
    model.set_rhs('z', dz)
    model.set_rhs('Velocity', vertcat(dVx, dVz))



    Pose_dot=vertcat( 
                            Velocity[0]*np.cos(Attitude[2])-Velocity[1]*np.sin(Attitude[2]),  
                            Velocity[1]*np.cos(Attitude[2])+Velocity[0]*np.sin(Attitude[2]), 
                            Velocity[2])
    Velocity_dot=vertcat( 
                            -inp[3]*(np.sin(Attitude[1]))/m,  
                            inp[3]*(np.sin(Attitude[0]))/m, 
                            -g+inp[3]*np.cos(Attitude[0])*np.cos(Attitude[1])/m
    )
    """
    Velocity_dot=vertcat( 
                            -inp[3]*(np.sin(Attitude[0])*np.sin(Attitude[2])+np.cos(Attitude[0])*np.cos(Attitude[2])*np.sin(Attitude[1]))/m,  
                            -inp[3]*(np.cos(Attitude[0])*np.sin(Attitude[2])*np.sin(Attitude[1])-np.cos(Attitude[2])*np.sin(Attitude[1]))/m, 
                            g-inp[3]*np.cos(Attitude[0])*np.cos(Attitude[1])/m
    )"""
    """
    Rate_dot = vertcat( 
                            inp[0]/Ixx ,  
                            inp[1]/Iyy , 
                            inp[2]/Izz 
    )
    """ 
    Rate_dot = vertcat( 
                            (Iyy-Izz)*Rate[1]*Rate[2]/Ixx + inp[0]/Ixx ,  
                            (Izz-Ixx)*Rate[0]*Rate[2]/Iyy + inp[1]/Iyy , 
                            (Ixx-Iyy)*Rate[1]*Rate[0]/Izz + inp[2]/Izz 
    ) 
    
    Attitude_dot=vertcat(Rate[0]+Rate[2]*(np.cos(Attitude[0])*np.tan(Attitude[1]))+Rate[1]*(np.sin(Attitude[0])*np.tan(Attitude[1])),
    Rate[1]*np.cos(Attitude[0])-Rate[2]*np.sin(Attitude[0]),
    Rate[2]*np.cos(Attitude[0])/np.cos(Attitude[1])+Rate[1]*np.sin(Attitude[0])/np.cos(Attitude[1]))
    """
    Attitude_dot=vertcat(Rate[0],
    Rate[1],
    Rate[2]
    )"""
    # Differential equations
    model.set_rhs('Pose', Pose_dot)
    model.set_rhs('Attitude', Attitude_dot)
    model.set_rhs('Rate', Rate_dot)
    model.set_rhs('Velocity', Velocity_dot)

    #ref_pose=[5,5,5]
    """
    current_pose=[Pose[0],Pose[1],Pose[2]]

    Err_pose=[ref_pose[0]-current_pose[0],ref_pose[1]-current_pose[1],ref_pose[2]-current_pose[2]]

    Distance_err=np.sqrt(pow(Err_pose[0],2)+pow(Err_pose[1],2)+pow(Err_pose[2],2))

    statu=if_else(Distance_err<1,True,False)

    ref_pose=if_else(statu==True,[0,0,0],[5,5,5])"""

    target_pose=ref_pose

    target_vel_body=[target_pose[0]-Pose[0],target_pose[1]-Pose[1],target_pose[2]-Pose[2]]

    

    target_vel=[target_vel_body[0]*np.cos(Attitude[2])+target_vel_body[1]*np.sin(Attitude[2]),target_vel_body[1]*np.cos(Attitude[2])-target_vel_body[0]*np.sin(Attitude[2]),target_vel_body[2]]
    
    for i in range(3):
        target_vel[i]*=3

    var_vel=norm_3d(target_vel)

    V_ratio=if_else(var_vel>const_vel,var_vel/const_vel,1)

    for i in range(3):
        target_vel[i]=target_vel[i]/V_ratio
    
    
    # Cost
    cost=.1*pow(1*target_vel[0]-Velocity[0],2)+.1*pow(1*target_vel[1]-Velocity[1],2)+.1*pow(1*target_vel[2]-Velocity[2],2)+5*pow(ref_pose[3]*np.pi-Attitude[2],2)

    model.set_expression('cost', cost)

    # Build the model
    model.setup()

    return model


