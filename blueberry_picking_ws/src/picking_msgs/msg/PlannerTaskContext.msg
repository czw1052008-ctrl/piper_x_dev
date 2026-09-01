# Discrete task context for trajectory planner (inputs, not outputs).

# fsm_state constants
uint8 FSM_IDLE=0
uint8 FSM_CLUSTER_ALIGN=1
uint8 FSM_APPROACH_FRUIT=2
uint8 FSM_RETRACT=3
uint8 FSM_HOLD=4

uint8 fsm_state
int32 cluster_id
int32 fruit_id
