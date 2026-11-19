Humanoid Tossing an Apple (MuJoCo Simulation)

In this project we made a MuJoCo simulation of a Unitree G1 humanoid robot tossing an apple from one hand to the other. Both palms face up. One hand throws the apple up, the other hand catches it and closes its fingers around it, and then the hands swap and the tossing continues.

Video (YouTube): https://youtu.be/aMpfSBcS8Bo

The robot model is not uploaded here because it is very large. It is taken from MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

Requirements:
Python 3 (we used Python 3.14) and VS Code.

How to run:

1) Download MuJoCo Menagerie from the link above (Code > Download ZIP) and extract it into a new folder.

2) Open that folder in VS Code and put live_toss.py in the same folder, next to the mujoco_menagerie folder. The folder should look like this:
live_toss.py
mujoco_menagerie (folder)

3) Open the terminal in VS Code and install the packages:
pip install mujoco numpy

4) Run the code:
python live_toss.py

The MuJoCo window opens and the robot starts tossing the apple. We can rotate the view with the mouse and zoom with the scroll wheel. Close the window to stop.

How the code works:
The robot's base is fixed and its legs stay in a standing pose, so only the arms move. The code finds which side of the hand is the palm, and then uses inverse kinematics to move each hand to a target position with the palm facing up. The throwing hand moves down a little and then quickly up, and the apple is released with a velocity calculated from projectile motion so that it lands in the other palm. The catching hand moves down a little to cushion it and its fingers close. MuJoCo solves the robot dynamics, M(q)q'' + C(q,q')q' + G(q) = tau, in the background at every time step.

The numbers at the top of live_toss.py (T_FLY, THROW_UP, GRIP, etc.) can be changed to make the toss higher or the grip tighter.


