# Closed BEV obstacle ring at constant z (base_link).
# xy is flattened [x0,y0,x1,y1,...]; n_verts = len(xy)/2, n_verts >= 3.
# label: 2=HARD 4=BERRY_ACTIVE 5=PLANT_SOFT 6=BERRY_PENDING
# constraint: polygon class for encoder + RViz (no pot class).
# is_floor: workspace bottom at z=0 (arm must not touch).
# cluster_id / berry_id: -1 if not applicable.
# track_id: stable polygon identity across frames; floor = -1.

uint8 CONSTRAINT_HARD=0
uint8 CONSTRAINT_SOFT=1
uint8 CONSTRAINT_BERRY_PENDING=2
uint8 CONSTRAINT_BERRY_ACTIVE=3

uint8 label
uint8 constraint
bool is_floor
int32 cluster_id
int32 berry_id
int32 track_id
float32 z_m
float32[] xy
