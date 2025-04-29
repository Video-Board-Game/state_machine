# ROS2 imports 
import rclpy
from rclpy.node import Node

# CV Bridge and message imports
from std_msgs.msg import String, Bool, Header
from geometry_msgs.msg import Pose2D, Point, PointStamped, PoseWithCovariance, PoseStamped
from builtin_interfaces.msg import Time
from std_srvs.srv import Empty

import numpy as np
from collections.abc import Callable
from collections import deque
import math
from enum import Enum
import time

from board_manipulator.BoardArmNode import MoveModes, Grasps
from arm_interfaces.msg import ArmStatus, ArmCommand

class State(Enum):
    IDLE = "IDLE"
    VALIDATE_MOVE = "VALIDATE_MOVE"
    MOVE_TO_START = "MOVE_TO_START"
    LOWER_FOR_GRASP = "LOWER_FOR_GRASP"
    SELECT_GRIPPER = "SELECT_GRIPPER"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT_FROM_START = "LIFT_FROM_START"
    MOVE_TO_GOAL = "MOVE_TO_GOAL"
    LOWER_FOR_RELEASE = "LOWER_FOR_RELEASE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    LIFT_FROM_GOAL = "LIFT_FROM_GOAL"
    STOW_ARM = "STOW_ARM"
    ERROR = "ERROR"
    
# launch using the: ros2 launch state_machine state_machine
# manually set state: ros2 topic pub -1 /set_state std_msgs/msg/String "{data: 'IDLE'}"

class StateManager(Node):

    def __init__(self):
        super().__init__('state_machine')
        
        ### TOPIC DECLARATION - ALL PARAMETERIZED THROUGH ROS2 LAUNCH

        ### DETECTION TOPICS --------------------------------------------------

        # Topic for sending selected object centroid to VBM
        # TODO not in use 
        self.declare_parameter('start_centroid_topic', 'start_centroid')
        self.declare_parameter('goal_centroid_topic', 'goal_centroid')

        # Topics for receiving 3D points from VBM
        self.declare_parameter('start_extract_topic', 'extract_start_centroid')
        self.declare_parameter('goal_extract_topic', 'extract_goal_centroid')

        ### ARM TOPICS --------------------------------------------------------
        # Topic for sending arm commands to move on trajectories
        self.declare_parameter('arm_command_topic', 'arm_command')

        # CustomArmMsg from Arm Node
        self.declare_parameter('arm_status_topic', 'arm_status')

        ### STATE TOPICS --------------------------------------------------
        
        # Topic to send state information
        self.declare_parameter('state_topic','state')

        # Topic to receive state commands
        self.declare_parameter('state_setter_topic', 'set_state')
        
        # -----
        # DEBUG
        self.declare_parameter('override_errors', False)
        self.declare_parameter('debug', 0b11111)
        self.override_errors = self.get_parameter('override_errors').value
        debug = self.get_parameter('debug').value
        self.debug_publish  = bool(debug & 0b10000)
        self.debug_xxxxx    = bool(debug & 0b01000)
        self.debug_detect   = bool(debug & 0b00100)
        self.debug_vbm      = bool(debug & 0b00010)
        self.debug_arm      = bool(debug & 0b00001)
        self.get_logger().info(f"Debug Flags: Publish: {self.debug_publish}, XXXXX: {self.debug_xxxxx},"+
                               f" Detection: {self.debug_detect}, VBM: {self.debug_vbm}, Arm: {self.debug_arm}")
        
        # INITIAL STATE
        self._state = State.IDLE
        self._received_state = State.IDLE
        self._start_extract_pt = PointStamped()
        self._goal_extract_pt = PointStamped()
        self._arm_status = ArmStatus()    

        # STORE PREVIOUS MESSAGE SENT PER TOPIC
        self.last_sent_messages = {}
        self.msg_timeout = 3

        # State machine progress variables
        self.gripper_selection = Grasps.OPEN
        self.gripper_threshold = 0.02 # Height above start piece in cm to switch gripper module
        self.movement_time = 1.0  # Default movement time in seconds
        self.approach_height = 0.05  # Height above target for approach
        self.grasp_tolerance = 0.01  # Position tolerance for grasping
        self.move_timeout = 5.0  # Timeout for arm movements
        self.move_mode = MoveModes.JOINT_SPACE
        self.last_operation_time = time.time()
        
        # SUBSCRIBERS
        self.state_setter_subscriber = self.create_subscription(
            String, 
            self.get_parameter('state_setter_topic').value, 
            self.receive_desired_state, 10)
        
        self.start_extract_subscriber = self.create_subscription(
            PointStamped,
            self.get_parameter('start_extract_topic').value,
            self.get_setter("start_extract_pt"), 10)
            
        self.goal_extract_subscriber = self.create_subscription(
            PointStamped,
            self.get_parameter('goal_extract_topic').value,
            self.get_setter("goal_extract_pt"), 10)

        self.arm_status_subscriber = self.create_subscription(
            ArmStatus, 
            self.get_parameter('arm_status_topic').value, 
            self.get_setter("arm_status"), 10)

        # PUBLISHERS            
        self.arm_command_publisher = self.create_publisher(
            ArmCommand, 
            self.get_parameter('arm_command_topic').value, 10)
            
        self.state_publisher = self.create_publisher(
            String,
            self.get_parameter('state_topic').value, 10)
             
        # Initialize tracking of new data from subscribers
        self.received_new = {
            self.state_setter_subscriber.topic: False,
            self.start_extract_subscriber.topic: False,
            self.goal_extract_subscriber.topic: False,
            self.arm_status_subscriber.topic: False,
        }  

        self.stow_joint_positions = [0.,0.,0.] # TODO
        self.unstow_joint_positions = [0.,0.,0.] # TODO
        self.current_pos = PoseStamped()
        self.current_pos.pose.position.x = self.stow_joint_positions[0]
        self.current_pos.pose.position.x = self.stow_joint_positions[1]
        self.current_pos.pose.position.x = self.stow_joint_positions[2]

        # initializes the state switching loop timer at the bottom of the file
        self.init_loop()


# ----------- PROPERTIES
    
    #### SET UP INCOMING MESSAGES AS PROPERTIES SO WE CAN KEEP TRACK OF WHAT
    #### NEW INFORMATION HAS / HASN'T BEEN ACCESSED
    
    # When python initializes properties, it hides the setters so that they get called when you do "self.property ="
    # This function retrieves the setter manually for any of the properties for the purposes of subscriber callbacks
    def get_setter(self, property):
        return lambda value: getattr(StateManager, property).fset(self, value)

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, value) -> State:
        if value not in State:
            raise ValueError(f"Invalid state: {value}")
        self._state = value

    @property
    def received_state(self) -> State:
        self.received_new[self.state_setter_subscriber.topic] = False
        return self._received_state

    @received_state.setter
    def received_state(self, value) -> State:
        if value not in State:
            raise ValueError(f"Invalid state: {value}")
        self._received_state = value
    
    def receive_desired_state(self, ros_msg: String) -> None:
        self.received_new[self.state_setter_subscriber.topic] = True
        try:
            string = ros_msg.data
            self.received_state = State(string)
            self.debug(True, f"State change queued from incoming message: {string, self._received_state}")
        except ValueError:
            self.debug(True, f"[WARNING] No matching state for string: {string}")

    @property
    def start_extract_pt(self) -> PointStamped:
        self.received_new[self.start_extract_subscriber.topic] = False
        return self._start_extract_pt
    
    @start_extract_pt.setter
    def start_extract_pt(self, ros_msg: PointStamped) -> None:
        self.received_new[self.start_extract_subscriber.topic] = True
        self._start_extract_pt = ros_msg
        
    @property
    def goal_extract_pt(self) -> PointStamped:
        self.received_new[self.goal_extract_subscriber.topic] = False
        return self._goal_extract_pt
    
    @goal_extract_pt.setter
    def goal_extract_pt(self, ros_msg: PointStamped) -> None:
        self.received_new[self.goal_extract_subscriber.topic] = True
        self._goal_extract_pt = ros_msg

    @property
    def arm_status(self) -> ArmStatus:
        self.received_new[self.arm_status_subscriber.topic] = False
        return self._arm_status
    
    @arm_status.setter
    def arm_status(self, ros_msg: ArmStatus):
        self.received_new[self.arm_status_subscriber.topic] = True
        self._arm_status = ros_msg

        self.current_pos = PoseStamped()
        self.current_pos.pose.position.x = ros_msg.ee_pos.position.x
        self.current_pos.pose.position.y = ros_msg.ee_pos.position.y
        self.current_pos.pose.position.z = ros_msg.ee_pos.position.z
# ----- HELPER FNs

    def publish_helper(self, publisher, message) -> None:
        """EXISTS SO WE DON'T PUBLISH DUPLICATE MESSAGES
        ONLY PASS IF THE MESSAGE IS THE SAME 
        ROS MESSAGES ARE SET UP TO BE EQUAL IF HEADERS+CONTENT ARE IDENTICAL"""

        if self.is_outgoing_new_msg(publisher, message):
            # DEBUG REPORTS ALL OUTGOING MESSAGES
            # Node name, topic, message, extra debug info (if present)
            self.debug(self.debug_publish, f"Topic - {publisher.topic}\tMessage - {message}")

            publisher.publish(message)
            self.last_sent_messages[publisher.topic] = {"msg": message, "time": time.time()}

    def is_outgoing_new_msg(self, publisher, message) -> bool:
        '''RETURNS TRUE IF THIS IS A NEW OUTGOING MESSAGE'''

        return not (publisher.topic in self.last_sent_messages 
                    and self.last_sent_messages[publisher.topic]["msg"] == message
                    # Additionally, send message if previous message is more than {self.msg_timeout} seconds old
                    and self.last_sent_messages[publisher.topic]["time"] > time.time() - self.msg_timeout)
    
    def is_new_data_from_subscriber(self, subscriber):
        '''RETURNS TRUE IF THERE WAS A NEW MESSAGE RECEIVED ON THIS SUBSCRIBER'''
        return self.received_new[subscriber.topic]

    def get_last_sent_message(self, publisher):
        '''
        returns last sent message under publisher topic
        if there is no last sent message w/ that topic, returns None
        '''

        if not publisher.topic in self.last_sent_messages:
            return None
            
        return self.last_sent_messages[publisher.topic]["msg"]

    def debug(self, if_debug:bool, string:String) -> None:
        '''
        if_debug is a boolean - if true, string gets printed to ros logger
        '''
        if if_debug:
            # Node name, topic, message, extra debug info (if present)
            self.get_logger().info(string)
            
    def is_timeout_expired(self, timeout_duration: float) -> bool:
        '''
        Returns True if more than timeout_duration seconds have passed since last_operation_time
        '''
        return (time.time() - self.last_operation_time) > timeout_duration

    def reset_timeout(self) -> None:
        '''
        Reset the operation timeout timer
        '''
        self.last_operation_time = time.time()

# ----- ARM SERVICE FUNCTIONS
    def stow_arm(self):
        """
        Send arm to stowed position
        """
        goal = PoseStamped()
        goal.pose.position.x = self.stow_joint_positions[0]
        goal.pose.position.y = self.stow_joint_positions[1]
        goal.pose.position.z = self.stow_joint_positions[2]
        self.sendArmCommand(goal)
    
    def unstow_arm(self):
        """
        Unstow the arm to prepare for movement
        """
        goal = PoseStamped()
        goal.pose.position.x = self.unstow_joint_positions[0]
        goal.pose.position.y = self.unstow_joint_positions[1]
        goal.pose.position.z = self.unstow_joint_positions[2]
        self.sendArmCommand(goal)

# ----- ARM CONTROL FUNCTIONS
    def openGripper(self): 
        '''send ROSmsg to arm control node to open gripper'''
        self.sendArmCommand(grasp_type=Grasps.OPEN)
    
    def closeGripper(self):
        '''send ROSmsg to arm control node to close gripper'''
        self.sendArmCommand(grasp_type=self.gripper_selection)
        
    def moveElevator(self, height: float):
        '''send ROSmsg to control elevator'''
        goal = self.current_pos
        goal.pose.position.z = height
        self.sendArmCommand(goal, move_type=MoveModes.ELEVATE)

    def sendArmCommand(self, goal: PoseStamped = None, movement_time: float = None, grasp_type: Grasps = Grasps.OPEN, 
                       move_type: MoveModes = MoveModes.JOINT_SPACE, tolerance: float = 0.02, alpha: float = 0.0,
                       grasp_at_end_of_movement: bool = False) -> None: 
        '''send ROSmsg to arm control node with a point, elevator position, and/or grasp position. reset timeout'''
        if goal is None:
            goal = self.current_pos

        if movement_time is None:
            movement_time = self.movement_time
            
        msg = ArmCommand()
        msg.goal.x = goal.pose.position.x
        msg.goal.y = goal.pose.position.y
        msg.goal.z = goal.pose.position.z
        msg.tolerance = tolerance # idk if this does anything
        msg.grasp_at_end_of_movement = grasp_at_end_of_movement # idk if this does anything
        msg.trajectory_mode = move_type.name
        msg.alpha = alpha # idk what this does
        msg.movement_time = movement_time
        msg.grasp_type = grasp_type.name
        self.publish_helper(self.arm_command_publisher, msg)
        self.reset_timeout()
        
    def calculatePositionDifference(self, target_pose: PoseStamped) -> float:
        '''
        Calculate the Euclidean distance between the current end effector position 
        and the target position
        '''
        diffx = self.arm_status.ee_pos.x - target_pose.pose.position.x
        diffy = self.arm_status.ee_pos.y - target_pose.pose.position.y 
        diffz = self.arm_status.ee_pos.z - target_pose.pose.position.z
        return math.sqrt(diffx**2 + diffy**2 + diffz**2)
        
    def isArmAtPosition(self, target_pose: PoseStamped) -> bool:
        '''
        Returns True if the arm is at the target position within tolerance
        '''
        return self.calculatePositionDifference(target_pose) <= self.grasp_tolerance
        
    def createElevatedPose(self, base_point: PointStamped) -> PoseStamped:
        '''
        Create a PoseStamped at the approach position above the target point
        '''
        elevated_pose = PoseStamped()
        elevated_pose.header = base_point.header
        elevated_pose.pose.position.x = base_point.point.x
        elevated_pose.pose.position.y = base_point.point.y
        elevated_pose.pose.position.z = base_point.point.z + self.approach_height #TODO
        return elevated_pose
        
    def createGraspPose(self, base_point: PointStamped) -> PoseStamped:
        '''
        Create a PoseStamped at the grasp position
        '''
        grasp_pose = PoseStamped()
        grasp_pose.header = base_point.header
        grasp_pose.pose.position.x = base_point.point.x
        grasp_pose.pose.position.y = base_point.point.y
        grasp_pose.pose.position.z = base_point.point.z
        return grasp_pose

# ----------- STATE FUNCTIONS
    
    def validateMove(self) -> State:
        '''
        Check if we have both start and goal points
        '''
        if not self.is_new_data_from_subscriber(self.start_extract_subscriber):
            self.debug(self.debug_vbm, "Waiting for start point data")
            return State.VALIDATE_MOVE
            
        if not self.is_new_data_from_subscriber(self.goal_extract_subscriber):
            self.debug(self.debug_vbm, "Waiting for goal point data")
            return State.VALIDATE_MOVE
            
        self.debug(self.debug_vbm, "Move validated, proceed to unstow and move to start")
        self.unstow_arm()
        return State.MOVE_TO_START
        
    def moveToStart(self) -> State:
        '''
        Move arm to position above the start position
        '''
        elevated_pose = self.createElevatedPose(self.start_extract_pt)
        self.sendArmCommand(elevated_pose, move_type=MoveModes.JOINT_SPACE)
        
        # Check if we've reached the position
        if self.isArmAtPosition(elevated_pose):
            self.debug(self.debug_arm, "Reached start position")
            return State.LOWER_FOR_GRASP
            
        # Check for timeout
        if self.is_timeout_expired(self.move_timeout):
            self.debug(self.debug_arm, "Timeout moving to start position")
            return State.ERROR
            
        return State.MOVE_TO_START
        
    def lowerForGrasp(self) -> State:
        '''
        Lower the arm to grasp position
        '''
        grasp_pose = self.createGraspPose(self.start_extract_pt)
        self.sendArmCommand(grasp_pose, move_type=MoveModes.JOINT_SPACE)
        
        # Check if we've reached the position
        if self.isArmAtPosition(grasp_pose):
            self.debug(self.debug_arm, "Lowered to grasp position")
            return State.SELECT_GRIPPER
            
        # Check for timeout
        if self.is_timeout_expired(self.move_timeout):
            self.debug(self.debug_arm, "Timeout lowering for grasp")
            return State.ERROR
            
        return State.LOWER_FOR_GRASP
        
    def selectGripperState(self) -> State:
        '''
        Select appropriate gripper based on detected object
        '''
        # This could be expanded to select between gripper types based on object properties
        # For now, we'll use the default or make a selection based on the z-depth
        
        # I.e. if object is game piece, use pinch gripper
        if self.start_extract_pt.point.z >= self.gripper_threshold:
            gripper_type = Grasps.PIECE
        else: # I.e. if object is card
            gripper_type = Grasps.CARD
            
        self.gripper_selection = gripper_type
        self.debug(self.debug_arm, f"Selected {gripper_type} gripper")
        return State.CLOSE_GRIPPER
        
    def closeGripperState(self) -> State:
        '''
        Close gripper to grasp object
        '''
        self.closeGripper()
        
        # Check if gripper is closed and grasping object
        if self.arm_status.grasping_object:
            self.debug(self.debug_arm, "Object grasped")
            return State.LIFT_FROM_START
            
        # Check for timeout
        if self.is_timeout_expired(2.0):  # shorter timeout for gripper operation
            self.debug(self.debug_arm, "Timeout closing gripper")
            return State.ERROR
            
        return State.CLOSE_GRIPPER
        
    def liftFromStart(self) -> State:
        '''
        Lift arm from start position with object
        '''
        elevated_pose = self.createElevatedPose(self.start_extract_pt)
        self.sendArmCommand(elevated_pose, move_type=MoveModes.ELEVATE)
        
        # Check if we've reached the elevated position
        if self.isArmAtPosition(elevated_pose):
            self.debug(self.debug_arm, "Lifted from start position")
            return State.MOVE_TO_GOAL
            
        # Check for timeout
        if self.is_timeout_expired(self.move_timeout):
            self.debug(self.debug_arm, "Timeout lifting from start")
            return State.ERROR
            
        return State.LIFT_FROM_START
        
    def moveToGoal(self) -> State:
        '''
        Move arm to position above goal position
        '''
        elevated_pose = self.createElevatedPose(self.goal_extract_pt)
        self.sendArmCommand(elevated_pose, move_type=MoveModes.JOINT_SPACE)
        
        # Check if we've reached the position
        if self.isArmAtPosition(elevated_pose):
            self.debug(self.debug_arm, "Reached goal position")
            return State.LOWER_FOR_RELEASE
            
        # Check for timeout
        if self.is_timeout_expired(self.move_timeout):
            self.debug(self.debug_arm, "Timeout moving to goal position")
            return State.ERROR
            
        return State.MOVE_TO_GOAL
        
    def lowerForRelease(self) -> State:
        '''
        Lower arm to release position
        '''
        release_pose = self.createGraspPose(self.goal_extract_pt)
        self.sendArmCommand(release_pose, move_type=MoveModes.ELEVATE)
        
        # Check if we've reached the position
        if self.isArmAtPosition(release_pose):
            self.debug(self.debug_arm, "Lowered to release position")
            return State.OPEN_GRIPPER
            
        # Check for timeout
        if self.is_timeout_expired(self.move_timeout):
            self.debug(self.debug_arm, "Timeout lowering for release")
            return State.ERROR
            
        return State.LOWER_FOR_RELEASE
        
    def openGripperState(self) -> State:
        '''
        Open gripper to release object
        '''
        self.openGripper()
        
        # Check if gripper is open
        if not self.arm_status.grasping_object:
            self.debug(self.debug_arm, "Object released")
            return State.LIFT_FROM_GOAL
            
        # Check for timeout
        if self.is_timeout_expired(2.0):  # shorter timeout for gripper operation
            self.debug(self.debug_arm, "Timeout opening gripper")
            return State.ERROR
            
        return State.OPEN_GRIPPER
        
    def liftFromGoal(self) -> State:
        '''
        Lift arm from goal position
        '''
        elevated_pose = self.createElevatedPose(self.goal_extract_pt)
        self.sendArmCommand(elevated_pose, move_type=MoveModes.ELEVATE)
        
        # Check if we've reached the elevated position
        if self.isArmAtPosition(elevated_pose):
            self.debug(self.debug_arm, "Lifted from goal position")
            return State.STOW_ARM
            
        # Check for timeout
        if self.is_timeout_expired(self.move_timeout):
            self.debug(self.debug_arm, "Timeout lifting from goal")
            return State.ERROR
            
        return State.LIFT_FROM_GOAL
        
    def stowArmState(self) -> State:
        '''
        Stow the arm
        '''
        self.stow_arm()
        
        # Check if arm is stowed
        if self.arm_status.is_stowed:
            self.debug(self.debug_arm, "Arm stowed")
            return State.IDLE
            
        # Check for timeout
        if self.is_timeout_expired(self.move_timeout):
            self.debug(self.debug_arm, "Timeout stowing arm")
            return State.ERROR
            
        return State.STOW_ARM
        
    def errorState(self) -> State:
        '''
        Handle error state
        '''
        self.debug(True, "ERROR STATE: Attempting to recover")
        
        # Try to stow the arm for safety
        self.stow_arm()
        
        # If configured to override errors, return to idle
        if self.override_errors:
            self.debug(True, "Error overridden, returning to IDLE")
            return State.IDLE
            
        # Stay in error state until manually reset
        return State.ERROR

# --- STATE MACHINE TRANSITION LOGIC

    def state_transitions(self, old_state, new_state):
        """
        Handle any special logic needed when transitioning between states
        """
        if old_state != new_state:
            self.debug(True, f"State transition: {old_state.value} -> {new_state.value}")
            # Reset timeout on state transition
            self.reset_timeout()
            
            # Handle specific transitions if needed
            if new_state == State.IDLE:
                # Make sure arm is stowed when returning to idle
                if not self.arm_status.is_stowed:
                    self.stow_arm()

# ----------- MAIN LOOP
    
    def main_loop(self):
        # Publish current state
        state_msg = String()
        state_msg.data = self.state.value
        self.publish_helper(self.state_publisher, state_msg)

        # Default to maintaining current state
        new_state = self.state

        # Execute state function based on current state
        if self.state == State.IDLE:
            # Idle state - wait for command
            pass
        elif self.state == State.VALIDATE_MOVE:
            new_state = self.validateMove()
        elif self.state == State.MOVE_TO_START:
            new_state = self.moveToStart()
        elif self.state == State.LOWER_FOR_GRASP:
            new_state = self.lowerForGrasp()
        elif self.state == State.SELECT_GRIPPER:
            new_state = self.selectGripperState()
        elif self.state == State.CLOSE_GRIPPER:
            new_state = self.closeGripperState()
        elif self.state == State.LIFT_FROM_START:
            new_state = self.liftFromStart()
        elif self.state == State.MOVE_TO_GOAL:
            new_state = self.moveToGoal()
        elif self.state == State.LOWER_FOR_RELEASE:
            new_state = self.lowerForRelease()
        elif self.state == State.OPEN_GRIPPER:
            new_state = self.openGripperState()
        elif self.state == State.LIFT_FROM_GOAL:
            new_state = self.liftFromGoal()
        elif self.state == State.STOW_ARM:
            new_state = self.stowArmState()
        elif self.state == State.ERROR:
            new_state = self.errorState()
            
        # Override with received state if there's an incoming state command
        if self.is_new_data_from_subscriber(self.state_setter_subscriber):
            self.debug(self.debug_publish, f"Overriding state from received message: {self.received_state}")
            new_state = self.received_state
        
        # Handle state transitions and update state
        self.state_transitions(self.state, new_state)     
        self.state = new_state

    def init_loop(self):
        '''
        Times the main loop to run at 20Hz (every 0.05 seconds)
        '''
        
        # Stow arm at node launch
        self.stow_arm() 

        self.timer = self.create_timer(0.05, self.main_loop)
        self.debug(True, "State Machine initialized with 20Hz timer")

def main(args=None):
    rclpy.init(args=args)

    manager_node = StateManager()
    
    # Now actually allow ROS to process callbacks
    rclpy.spin(manager_node)

    manager_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()