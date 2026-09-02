# 2D instance contour in the source image (P1).
# class_id: 1=berry 2=branch 3=rigid (same as scene_seg semantic).
# polygon_uv: flattened [u0,v0,u1,v1,...] in pixels, n_verts = len/2, n>=3.
# bbox_uv: [u0,v0,u1,v1].

int32 id
uint8 class_id
float32 score
float32[] polygon_uv
float32[4] bbox_uv
int32 area_px
