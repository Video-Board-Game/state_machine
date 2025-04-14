# ROS2 imports 
import rclpy
from rclpy.node import Node

# CV Bridge and message imports
from std_msgs.msg import String, Bool, Header
from interfaces.msg import ArmStatus, ArmCommand
from geometry_msgs.msg import Pose2D, Point, PointStamped, PoseWithCovariance, PoseStamped
from builtin_interfaces.msg import Time
from std_srvs.srv import Empty


import numpy as np
from collections.abc import Callable
from collections import deque
import math
from enum import Enum
import time

class State(Enum):
    STOW = "STOW"
    MOVE_ARM = "MOVE_ARM" # move manipulator from pt to pt
    MOVE_GRIPPER = "MOVE_GRIPPER" # lower gripper to depth, open/close, raise gripper
    
# launch using the: ros2 launch state_machine state_machine
# manually set state: ros2 topic pub -1 /set_state std_msgs/msg/String "{data: 'SEARCH'}"

class StateManager(Node):

    def __init__(self):
        super().__init__('state_machine')
        
        ### TOPIC DECLARATION - ALL PARAMETERIZED THROUGH ROS2 LAUNCH

        ### DETECTION TOPICS --------------------------------------------------

        # Topic for sending selected object centroid to VBM TODO start, goal centroids
        self.declare_parameter('centroid_topic', 'selected_centroid')

        # Topic for receiving 3D point from VBM
        self.declare_parameter('vbm_extract_topic', 'extract_centroid')

        ### ARM TOPICS --------------------------------------------------------

        # Topic for sending gripper to move
        self.declare_parameter('grasp_command_topic', 'force_grasp')

        # Topic for sending arm commands to move on trajectories and potentiall to grasp
        self.declare_parameter('arm_command_topic', 'move_arm_command')

        # CustomArmMsg from Arm Node
        self.declare_parameter('arm_status_topic', 'arm_status')

        # Topic for Stow Service Call to Arm Node
        self.declare_parameter('arm_stow_service_topic', 'stow_arm')

        # Topic for Unstow Service Call to Arm Node
        self.declare_parameter('arm_unstow_service_topic', 'unstow_arm') # TODO check if needed

        ### STATE TOPICS --------------------------------------------------
        
        # Topic to send state information
        self.declare_parameter('state_topic','state')

        self.declare_parameter('state_setter_topic', 'set_state')
        
        # -----
        # SERVICES

        # self.in_range_service = self.create_client(Bool, self.get_parameter('arm_grasp_topic').value) # TODO fix or delete

        # -----
        # DEBUG
        self.declare_parameter('override_errors',False)
        self.declare_parameter('debug', 0b11111)
        self.override_errors = self.get_parameter('override_errors').value
        debug = self.get_parameter('debug').value
        self.debug_publish  = bool(debug & 0b10000)
        # self.debug_xxxxx    = bool(debug & 0b01000)
        # self.debug_detect   = bool(debug & 0b00100)
        self.debug_vbm      = bool(debug & 0b00010)
        self.debug_arm      = bool(debug & 0b00001)
        self.get_logger().info(f"Debug Flags: Publish: {self.debug_publish}, XXXXX {self.debug_xxxxx},"+
                               f" Detection {self.debug_detect}, VBM {self.debug_vbm}, Arm {self.debug_arm}")
        # INITIAL STATE
        self._state = State.STOW
        self._received_state = State.STOW
        self._extract_pt = PointStamped()
        self._arm_status = ArmStatus()    

        # STORE PREVIOUS MESSAGE SENT PER TOPIC
        self.last_sent_messages = {}
        self.msg_timeout = 3

        # waiting, stowing, unstowing state variables
        self.next_state = State.MOVE_ARM
        self.start = True # whether moving to start (True) or goal (False), toggles after move_arm

        # SUBSCRIBERS

        self.state_setter_subscriber = self.create_subscription(String, 
                                            self.get_parameter('state_setter_topic').value, 
                                            self.receive_desired_state, 10)
        # TODO start + goal pts
        self.extract_subscriber = self.create_subscription(PointStamped,
                                                    self.get_parameter('vbm_extract_topic').value,
                                                    self.get_setter("extract_pt"), 10)

        self.arm_status_subscriber = self.create_subscription(ArmStatus, 
                                                    self.get_parameter('arm_status_topic').value, 
                                                    self.get_setter("arm_status"), 10)

        # STOW ARM SERVICE CLIENT
        self.stow_arm_client = self.create_client(Empty, self.get_parameter('arm_stow_service_topic').value)
        self.unstow_arm_client = self.create_client(Empty, self.get_parameter('arm_unstow_service_topic').value)

        # PUBLISHERS
        self.force_grasp_publisher = self.create_publisher(Bool, 
                                                    self.get_parameter('grasp_command_topic').value, 10)
        self.arm_command_publisher = self.create_publisher(ArmCommand, 
                                                    self.get_parameter('arm_command_topic').value, 10)
        self.state_publisher = self.create_publisher(String,
                                                    self.get_parameter('state_topic').value, 10)
                
        # new needs to exist for the properties but needs the subscribers to exist as well
        self.received_new = {
            self.state_setter_subscriber.topic: False,
            self.extract_subscriber.topic: False,
            self.arm_status_subscriber.topic: False,
        }  

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
            self.debug(True, f"State changed queued from incoming message: {string, self._received_state}")
        except ValueError:
            self.debug(True, f"[WARNING] No matching state for string: {string}")

    @property
    def extract_pt(self) -> PointStamped:
        self.received_new[self.extract_subscriber.topic] = False
        return self._extract_pt
    
    @extract_pt.setter
    def extract_pt(self, ros_msg: PointStamped) -> None:
        self.received_new[self.extract_subscriber.topic] = True
        self._extract_pt = ros_msg

    @property
    def arm_status(self) -> ArmStatus:
        self.received_new[self.arm_status_subscriber.topic] = False
        return self._arm_status
    
    @arm_status.setter
    def arm_status(self, ros_msg: ArmStatus):
        self.received_new[self.arm_status_subscriber.topic] = True
        self._arm_status = ros_msg

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

# ----- STOW FNs
    def stow_arm(self):
        # Call the stow_arm service
        self.get_logger().info('Calling stow_arm service...')
        request = Empty.Request()
        future = self.stow_arm_client.call_async(request)
        # Internal function for service debug messages
        def service_debug_message(response):
            try:
                response.result()
                return "Stow Service Success"
            except:
                return "Stow Service Failure"

        future.add_done_callback(lambda response: self.debug(self.debug_arm, service_debug_message(response)))
    
    def unstow_arm(self):
        # Call the unstow_arm service
        self.get_logger().info('Calling unstow_arm service...')
        request = Empty.Request()
        future = self.unstow_arm_client.call_async(request)
        # Internal function for service debug messages
        def service_debug_message(response):
            try:
                response.result()
                return "Unstow Service Success"
            except:
                return "Unstow Service Failure"

        future.add_done_callback(lambda response: self.debug(self.debug_arm, service_debug_message(response)))

# ----- ARM HELPERS
    # TODO select gripper mode
    def openGripper(self):
        '''send ROSmsg to arm control node to open gripper'''
        self.publish_helper(self.force_grasp_publisher, Bool(data=True))

    
    def closeGripper(self):
        '''send ROSmsg to arm control node to close gripper'''
        self.publish_helper(self.force_grasp_publisher, Bool(data=False))


    def sendArmCommand(self, poseStampedMsg:PoseStamped, task_space:str="track", 
                       grasp_at_end_of_movement:bool=False, movement_time:float=1.0) -> None:
        '''send ROSmsg to arm control node with a point'''
        msg = ArmCommand()
        msg.goal.x = poseStampedMsg.pose.position.x
        msg.goal.y = poseStampedMsg.pose.position.y
        msg.goal.z = poseStampedMsg.pose.position.z
        msg.tolerance = 0.05 # meters, this is the default tolerance for arm movement
        msg.grasp_at_end_of_movement = grasp_at_end_of_movement # use the parameter for grasping
        msg.trajectory_mode = task_space # task space or joint space 
        msg.movement_time = movement_time # seconds to complete the movement, default is 1.0s
        self.publish_helper(self.arm_command_publisher,msg) # publish the poseStamped to the arm command topic
    
# ----------- STATE FUNCTIONS
# ----- MOVE_ARM
    def move_arm(self) -> State:
        '''
        this loops when move_arm is current state
        no input, outputs State
        '''
        # calculate grasp
        # generate posestamped message from grasp    
        
        pt_subscriber = self.extract_subscriber if self.start else self.extract_subscriber # TODO check that start and end points received somewhere in state machine

        
        if self.is_new_data_from_subscriber(pt_subscriber): 
            point = self.extract_pt if self.start else self.extract_pt # TODO
            pt = PoseStamped()
            pt.header = point.header
            pt.pose.position = point.point
             
            diffx = self.arm_status.ee_pos.x - pt.pose.position.x
            diffy = self.arm_status.ee_pos.y - pt.pose.position.y 
            diffz = self.arm_status.ee_pos.z - pt.pose.position.z
            posDiff = math.sqrt(diffx**2 + diffy**2 + diffz**2)
            
            # send the grasp command to the arm
            self.sendArmCommand(pt, task_space="track", grasp_at_end_of_movement=True)

            self.debug(self.debug_arm, f"Position difference between armPos and setpoint: {posDiff}")
        
            if self.arm_status.grasping_object == True:
                self.debug(self.debug_arm, f'grasp successful, moving to next state')
                return State.MOVE_GRIPPER
        
        return State.MOVE_ARM 

# ----- MOVE_GRIPPER
    def move_gripper(self) -> State:
        '''
        this loops when move_gripper is current state
        no input, outputs State
        '''
        # lower gripper, open/close, raise gripper
        
        pt_subscriber = self.extract_subscriber if self.start else self.extract_subscriber # TODO check that start and end points received somewhere in state machine

        
        if self.is_new_data_from_subscriber(pt_subscriber): 
            point = self.extract_pt if self.start else self.extract_pt # TODO
            pt = PoseStamped()
            pt.header = point.header
            pt.pose.position = point.point
             
            diffx = self.arm_status.ee_pos.x - pt.pose.position.x
            diffy = self.arm_status.ee_pos.y - pt.pose.position.y 
            diffz = self.arm_status.ee_pos.z - pt.pose.position.z
            posDiff = math.sqrt(diffx**2 + diffy**2 + diffz**2)
            
            # send the grasp command to the arm
            # self.sendArmCommand(pt, task_space="track", grasp_at_end_of_movement=True)
            # sendElevatorCommand(lower if self.start else raise) # TODO implement raise/lower function that returns whether reached position status
            if self.arm_status.elevator == True: # TODO implement status for elevator reaching a goal
                self.closeGripper() if self.start else self.openGripper() # TODO implement return status and switch grippers depending on depth
                if self.arm_status.gripper == True:
                    # self.sendElevatorCommand(raise if self.start else lower) # TODO implement 
                    pass


            self.debug(self.debug_arm, f"Position difference between armPos and setpoint: {posDiff}")
        
            if self.arm_status.grasping_object == True: # TODO change to elevator/gripper routine finished
                self.debug(self.debug_arm, f'grasp successful, moving to next state')
                return State.MOVE_ARM if self.start else State.STOW
        
        return State.MOVE_GRIPPER

# --- STATE MACHINE TRANSITION LOGIC

    def state_transitions(self, old_state, new_state):
        # TODO implement or remove
        pass
            
            

# ----------- MAIN LOOP
    
    def main_loop(self):
        state_msg = String()
        state_msg.data = self.state.value
        self.publish_helper(self.state_publisher, state_msg)

        new_state = self.state

        if self.state == State.STOW:
            new_state = self.stow()
        elif self.state == State.MOVE_ARM:
            new_state = self.move_arm(self.start)
        elif self.state == State.MOVE_GRIPPER:
            new_state = self.move_gripper(self.start)
            
        if self.is_new_data_from_subscriber(self.state_setter_subscriber):
            # Use new state from message if there's an incoming state
            self.debug(self.debug_publish,f"Updating state from received message: {self.received_state}")
            new_state = self.received_state
        
        # any state transition behavior and set state
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