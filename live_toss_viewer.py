import os, time, mujoco, mujoco.viewer, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

 
T_FLY = 0.50           
THROW_UP = 0.10        
APPLE_OFFSET = 0.05   
PALM_SHIFT = -0.03    
FINGER_TILT = 0.2      
CURL = 0.0             
GRIP = 0.6             
T_GRIP = 0.15          
 
REST_X, REST_Y, REST_Z = 0.32, 0.20, 0.14    
WIND_DOWN, DIP = 0.06, 0.05
T_WIND, T_REL, T_RET, T_CUSH = 0.35, 0.60, 0.70, 0.40
T_CYC = T_REL + T_FLY + T_CUSH + 0.2

G1_DIR = None
for root, dirs, files in os.walk(HERE):
    if os.path.basename(root) == "unitree_g1" and "g1.xml" in files:
        G1_DIR = root
        break
if G1_DIR is None:
    raise SystemExit("unitree_g1 folder not found. Put mujoco_menagerie next to this file.")

robot_xml = "g1_with_hands.xml" if os.path.exists(os.path.join(G1_DIR, "g1_with_hands.xml")) else "g1.xml"
print("Using robot file:", robot_xml)

SCENE = os.path.join(G1_DIR, "apple_scene_auto.xml")
with open(SCENE, "w") as f:
    f.write(f"""<mujoco model="unitree_apple_toss">
  <include file="{robot_xml}"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <quality offsamples="8" shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".15 .2 .25" rgb2=".25 .3 .35"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <body name="apple" pos="0.28 -0.22 0.85">
      <freejoint name="apple_free"/>
      <geom name="apple_geom" type="sphere" size="0.038" mass="0.15" rgba="0.9 0.1 0.1 1" friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
""")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
dk = mujoco.MjData(model)      

def act(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
def jnt(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
def body(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
def bname(b): return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)

aj = jnt("apple_free")
adof, aqpos = model.jnt_dofadr[aj], model.jnt_qposadr[aj]
ageom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
model.geom_contype[ageom] = 0           
model.geom_conaffinity[ageom] = 0

 
STAND = {"hip_pitch_joint": -0.312, "knee_joint": 0.669, "ankle_pitch_joint": -0.363}
for side in ("left", "right"):
    for name, q in STAND.items():
        j, a = jnt(f"{side}_{name}"), act(f"{side}_{name}")
        if j >= 0: data.qpos[model.jnt_qposadr[j]] = q
        if a >= 0: data.ctrl[a] = q
PELVIS_Z = 0.79
data.qpos[0:7] = [0.0, 0.0, PELVIS_Z, 1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)
pelvis_pos = data.xpos[body("pelvis")].copy()

 
FINGERS = {"left": [], "right": []}     
for a in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
    for side in ("left", "right"):
        if name.startswith(f"{side}_hand_") and "thumb_0" not in name:
            lo, hi = model.actuator_ctrlrange[a]
            if abs(lo) < 1e-6 and abs(hi) > 1e-6:
                FINGERS[side].append((a, hi))
            elif abs(hi) < 1e-6 and abs(lo) > 1e-6:
                FINGERS[side].append((a, lo))

def set_fingers(levels):
    for side in ("left", "right"):
        for a, end in FINGERS[side]:
            data.ctrl[a] = end * levels[side]

 
ARM_J = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
ARM = {}
for side in ("left", "right"):
    js = [jnt(f"{side}_{n}_joint") for n in ARM_J]
    lo = np.array([model.jnt_range[j][0] if model.jnt_limited[j] else -3.0 for j in js])
    hi = np.array([model.jnt_range[j][1] if model.jnt_limited[j] else 3.0 for j in js])
    sign = 1.0 if side == "left" else -1.0
    ARM[side] = dict(
        qadr=np.array([model.jnt_qposadr[j] for j in js]),
        dadr=np.array([model.jnt_dofadr[j] for j in js]),
        act=[act(f"{side}_{n}_joint") for n in ARM_J],
        lo=lo, hi=hi,
        q_home=np.array([-0.35, 0.15 * sign, 0.0, 0.55, 0.0, 0.0, 0.0]),
    )

WR = {s: body(f"{s}_wrist_yaw_link") for s in ("left", "right")}

 
def descendants(root):
    out = []
    for b in range(model.nbody):
        p = model.body_parentid[b]
        while p != 0:
            if p == root:
                out.append(b); break
            p = model.body_parentid[p]
    return out

CAL = {}
for side in ("left", "right"):
    w = WR[side]
    desc = descendants(w)
    cand = [b for b in desc if ("index" in (bname(b) or "") or "middle" in (bname(b) or ""))] or desc
    dk.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, dk)
    wpos = dk.xpos[w].copy()
    R = dk.xmat[w].reshape(3, 3).copy()
    open_all = np.array([dk.xpos[b] for b in desc])
    open_c = np.array([dk.xpos[b] for b in cand])
    for a, end in FINGERS[side]:
        j = model.actuator_trnid[a, 0]
        dk.qpos[model.jnt_qposadr[j]] = 0.6 * end
    mujoco.mj_kinematics(model, dk)
    closed_c = np.array([dk.xpos[b] for b in cand])
    disp = closed_c - open_c
    top = np.argsort(np.linalg.norm(disp, axis=1))[-3:]
    fd = open_c[top].mean(axis=0) - wpos
    fd /= (np.linalg.norm(fd) + 1e-9)
    curl = disp[top].mean(axis=0)
    curl = curl - np.dot(curl, fd) * fd
    n = curl / (np.linalg.norm(curl) + 1e-9)
    centroid = open_all.mean(axis=0)
    point = centroid + n * APPLE_OFFSET + fd * PALM_SHIFT
    CAL[side] = dict(h=R.T @ (point - wpos), n=R.T @ n, f=R.T @ fd)
    print(side, ": finger bodies used:", [bname(cand[i]) for i in top])

N_T = np.array([0.0, 0.0, 1.0])                          
F_T = np.array([1.0, 0.0, FINGER_TILT]); F_T /= np.linalg.norm(F_T)    
ORI_W = 0.2

def frame(dd, side):
    w = WR[side]
    R = dd.xmat[w].reshape(3, 3)
    c = CAL[side]
    return dd.xpos[w] + R @ c["h"], R @ c["n"], R @ c["f"]

def hold_point(side):
    return frame(data, side)[0]

def ik_solve(side, target, q0, iters=15):
    A = ARM[side]
    q = q0.copy()
    dk.qpos[:] = data.qpos
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        dk.qpos[A["qadr"]] = q
        mujoco.mj_kinematics(model, dk)
        mujoco.mj_comPos(model, dk)
        pt, n_w, f_w = frame(dk, side)
        e = np.concatenate([target - pt, ORI_W * (np.cross(n_w, N_T) + np.cross(f_w, F_T))])
        mujoco.mj_jac(model, dk, jacp, jacr, pt, WR[side])
        J = np.vstack([jacp[:, A["dadr"]], ORI_W * jacr[:, A["dadr"]]])
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
        dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.1 * (A["q_home"] - q))
        q = np.clip(q + 0.6 * dq, A["lo"], A["hi"])
    return q

def rest_point(side):
    return pelvis_pos + np.array([REST_X, REST_Y if side == "left" else -REST_Y, REST_Z])

 
QSOL = {}
for side in ("left", "right"):
    q = ik_solve(side, rest_point(side), ARM[side]["q_home"], iters=300)
    QSOL[side] = q
    data.qpos[ARM[side]["qadr"]] = q
    for a, qa in zip(ARM[side]["act"], q):
        if a >= 0: data.ctrl[a] = qa
mujoco.mj_forward(model, data)
for side in ("left", "right"):
    pt, n_w, f_w = frame(data, side)
    print(side, ": hand position error (m):", round(float(np.linalg.norm(pt - rest_point(side))), 3),
          "| palm-up alignment (1 = perfect):", round(float(n_w @ N_T), 2),
          "| fingers-forward alignment:", round(float(f_w @ F_T), 2))

 
def mj(u):
    u = min(max(u, 0.0), 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5

UP = np.array([0.0, 0.0, 1.0])
OTHER = {"right": "left", "left": "right"}

def targets(tl, thrower):
    c = OTHER[thrower]
    rt, rc = rest_point(thrower), rest_point(c)
    if tl < T_WIND:
        pt = rt - UP * WIND_DOWN * mj(tl / T_WIND)
    elif tl < T_REL:
        pt = rt + UP * (-WIND_DOWN + (WIND_DOWN + THROW_UP) * mj((tl - T_WIND) / (T_REL - T_WIND)))
    else:
        pt = rt + UP * THROW_UP * (1.0 - mj((tl - T_REL) / T_RET))
    t_catch = T_REL + T_FLY
    if tl < t_catch:
        pc = rc
    else:
        pc = rc - UP * DIP * np.sin(np.pi * min((tl - t_catch) / T_CUSH, 1.0))
    return pt, pc

def grip_levels(tl, thrower):
    c = OTHER[thrower]
    t_catch = T_REL + T_FLY
     
    if tl < T_WIND:
        lt = GRIP
    elif tl < T_REL:
        lt = CURL + (GRIP - CURL) * (1.0 - mj((tl - T_WIND) / (T_REL - T_WIND)))
    else:
        lt = CURL
     
    if tl < t_catch:
        lc = CURL
    else:
        lc = CURL + (GRIP - CURL) * mj((tl - t_catch) / T_GRIP)
    return {thrower: lt, c: lc}

dt = model.opt.timestep
G = np.array([0.0, 0.0, -9.81])
ctrl_every = max(1, int(round(0.01 / dt)))
step = 0
held = "right"

print("Opening the MuJoCo window... Close the window to stop.")
viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.lookat[:] = [0.15, 0.0, 0.95]
viewer.cam.distance = 1.8
viewer.cam.azimuth = 145
viewer.cam.elevation = -10

wall0 = time.time()
while viewer.is_running():
    data.qpos[0:7] = [0.0, 0.0, PELVIS_Z, 1.0, 0.0, 0.0, 0.0]; data.qvel[0:6] = 0.0
    mujoco.mj_kinematics(model, data)

    k = int(data.time // T_CYC)
    tl = data.time - k * T_CYC
    thrower = "right" if k % 2 == 0 else "left"
    catcher = OTHER[thrower]

    if held == thrower and tl >= T_REL:                      
        p0 = hold_point(thrower)
        p1 = hold_point(catcher)
        data.qpos[aqpos:aqpos+3] = p0
        data.qvel[adof:adof+3] = (p1 - p0) / T_FLY - 0.5 * G * T_FLY
        data.qvel[adof+3:adof+6] = 0.0
        held = None
    elif held is None and tl >= T_REL + T_FLY:                 
        held = catcher

    if held is not None:
        data.qpos[aqpos:aqpos+3] = hold_point(held)
        data.qpos[aqpos+3:aqpos+7] = [1, 0, 0, 0]
        data.qvel[adof:adof+6] = 0.0

    if step % ctrl_every == 0:
        pt, pc = targets(tl, thrower)
        for side, tgt in ((thrower, pt), (catcher, pc)):
            QSOL[side] = ik_solve(side, tgt, QSOL[side])
            for a, qa in zip(ARM[side]["act"], QSOL[side]):
                if a >= 0: data.ctrl[a] = qa
    set_fingers(grip_levels(tl, thrower))

    mujoco.mj_step(model, data)
    step += 1

    if step % 8 == 0:                        
        viewer.sync()
        ahead = data.time - (time.time() - wall0)
        if ahead > 0:
            time.sleep(ahead)

viewer.close()