"""
Simply copy and paste this into a Maya script, and select two cameras
It will generate lines/arcs to visualize the yaw-invariant angle between their
bisector and nadir (toward Earth center).
"""


import math
import maya.cmds as mc
import maya.api.OpenMaya as om

# Validate selection
sel = mc.ls(sl=True, type='transform')
if len(sel) != 2:
    raise RuntimeError("Select exactly two camera transforms (the pair).")
cam1, cam2 = sel[0], sel[1]

def forward(cam):
    m = om.MMatrix(mc.xform(cam, q=True, ws=True, m=True))
    q = om.MTransformationMatrix(m).rotation(om.MEulerRotation.kXYZ).asQuaternion()
    v = om.MVector(0,0,-1).rotateBy(q)  # Maya camera forward = local -Z
    try: v.normalize()
    except: pass
    return v

def pos(cam):
    t = mc.xform(cam, q=True, ws=True, t=True)
    return om.MVector(float(t[0]), float(t[1]), float(t[2]))

def angle_deg(a, b):
    try:
        a = om.MVector(a.x, a.y, a.z); a.normalize()
    except: a = om.MVector(1,0,0)
    try:
        b = om.MVector(b.x, b.y, b.z); b.normalize()
    except: b = om.MVector(1,0,0)
    dot = max(-1.0, min(1.0, a.x*b.x + a.y*b.y + a.z*b.z))
    return math.degrees(math.acos(dot))

def set_curve_color(node, rgb):
    shapes = mc.listRelatives(node, shapes=True, fullPath=True) or []
    for s in shapes:
        try:
            mc.setAttr(s + ".overrideEnabled", 1)
            mc.setAttr(s + ".overrideRGBColors", 1)
            mc.setAttr(s + ".overrideColorRGB", float(rgb[0]), float(rgb[1]), float(rgb[2]))
        except:
            pass

# Compute bisector (acute) and nadir
f1 = forward(cam1)
f2 = forward(cam2)
f = om.MVector(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z)  # sum of unit forwards -> acute bisector
try: f.normalize()
except: pass
p = pos(cam1)  # both cams share the same root position
n = om.MVector(-p.x, -p.y, -p.z)  # nadir toward Earth center (origin)
try: n.normalize()
except: pass

# Angle
ang = angle_deg(f, n)
print("yaw_invarient_degrees =", ang)

# Draw rays
scale = 200.0  # adjust for scene scale
p0 = (p.x, p.y, p.z)
pb = (p.x + f.x*scale, p.y + f.y*scale, p.z + f.z*scale)
pn = (p.x + n.x*scale, p.y + n.y*scale, p.z + n.z*scale)

bis_curve = mc.curve(d=1, p=[p0, pb], name="bisector_line#")
nad_curve = mc.curve(d=1, p=[p0, pn], name="nadir_line#")
set_curve_color(bis_curve, (1.0, 1.0, 0.0))  # yellow
set_curve_color(nad_curve, (0.0, 1.0, 1.0))  # cyan

# Draw arc between bisector and nadir within their plane
# Use decomposition: n = cos(theta) * u + sin(theta) * v_unit, where u = f, v_unit is n's component perpendicular to u
try:
    u = om.MVector(f.x, f.y, f.z); u.normalize()
    # Component of n perpendicular to u
    proj = om.MVector(n.x - (n*u)*u.x, n.y - (n*u)*u.y, n.z - (n*u)*u.z)
    if proj.length() < 1e-8:
        arc_curve = None  # vectors are colinear; skip arc
    else:
        v_unit = om.MVector(proj.x, proj.y, proj.z); v_unit.normalize()
        theta = math.radians(ang)
        r = 0.6 * scale
        steps = max(16, int(max(10, ang/5.0)))
        pts = []
        for k in range(steps+1):
            t = (k/float(steps)) * theta
            dir_vec = om.MVector(u.x*math.cos(t) + v_unit.x*math.sin(t),
                                 u.y*math.cos(t) + v_unit.y*math.sin(t),
                                 u.z*math.cos(t) + v_unit.z*math.sin(t))
            pts.append((p.x + dir_vec.x*r, p.y + dir_vec.y*r, p.z + dir_vec.z*r))
        arc_curve = mc.curve(d=1, p=pts, name="yawInv_arc#")
        set_curve_color(arc_curve, (1.0, 0.0, 1.0))  # magenta

        # Angle label at arc midpoint
        mid = pts[len(pts)//2]
        try:
            txt = mc.textCurves(t=f"{round(ang, 2)}°", ch=False, name="yawInv_label#")[0]
            mc.xform(txt, ws=True, t=mid)
        except:
            pass
except:
    pass