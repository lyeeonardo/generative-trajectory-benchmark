#!/usr/bin/env python3
"""Render only the current environment figure from recorded successful runs.

Rendering changes are cosmetic. This script never trains, queries a policy,
steps dynamics, or changes the source dataset.
"""
from pathlib import Path
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from environment.task import UphillTask, scene
from environment.success import score_trace
from envs.mujoco_tilted_board import board_rotation, _xml_for_scene, MujocoRigidState
from scripts.paper_figure_style import apply_style

OUT = ROOT / "results/paper/figures"
SOURCE = OUT / "environment_source"
# Same route and ordinal in all three conditions; no trajectory cherry-picking.
PANELS = [(3, -15, 10), (4, 0, 0), (5, 15, -10)]
RESOLUTION = 1600
ELEVATION_DEGREES = 35.0
DISTANCE = 4.65
TARGET = np.array([0., 0., -.10])

def camera_pose(offset):
    az,el=np.deg2rad([offset,ELEVATION_DEGREES])
    pos=TARGET + DISTANCE*np.array([np.sin(az)*np.cos(el),-np.cos(az)*np.cos(el),np.sin(el)])
    forward=(TARGET-pos)/np.linalg.norm(TARGET-pos)
    right=np.cross(forward,np.array([0.,0.,1.]));right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    return pos,np.r_[right,up]

def as_xml_numbers(values):
    return " ".join(f"{v:.12g}" for v in values)

def render_panel(case,lateral,view_offset):
    path=ROOT/f"datasets/uphill_push_v1/episodes/train/c{case}_side-1/episode_00.npz"
    before=(path.stat().st_size,path.stat().st_mtime_ns)
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    meta=json.loads(path.with_suffix(".json").read_text())
    assert meta["case_id"]==case and meta["side"]==-1 and meta["obstacle_x"]==0
    safe=~(d["events"][:,1] | d["events"][:,2])
    outcome=score_trace(d["observations"],np.array([0.,.84]),safe)
    assert outcome["success"],f"Reference {meta['episode_id']} did not succeed"
    end=int(outcome["success_step"])
    assert safe[:end].all()
    for t in [end-1,end]:
        assert np.linalg.norm(d["observations"][t,:2]-[0.,.84])<=.12
        assert np.linalg.norm(d["observations"][t,2:4])<.15
    spec=scene(0,lateral)
    env=UphillTask();env.reset(spec)
    R=board_rotation(spec)
    position,axes=camera_pose(view_offset)
    xml=_xml_for_scene(spec,env.config,env.physics_config)
    xml=xml.replace("<worldbody>",'<asset><texture name="white_background" type="skybox" builtin="flat" width="32" height="32" rgb1="1 1 1" rgb2="1 1 1"/></asset><worldbody>',1)
    # An identical gray plane supplies the shared display floor and real shadows.
    # The plane is non-contact geometry added only to the rendering model.
    xml=xml.replace("<worldbody>",'<worldbody><geom name="display_ground" type="plane" pos="0 0 -0.62" size="0 0 0.025" rgba="0.72 0.72 0.72 1" contype="0" conaffinity="0"/>',1)
    camera=f'<camera name="orbit" mode="fixed" pos="{as_xml_numbers(position)}" xyaxes="{as_xml_numbers(axes)}" fovy="42"/>'
    xml=re.sub(r'<camera name="orbit"[^>]*/>',camera,xml)
    xml=xml.replace('<light name="key_light"','<light name="key_light" directional="true"')
    env.model=mujoco.MjModel.from_xml_string(xml)
    env.data=mujoco.MjData(env.model);env._cache_ids()
    pose_index = round(end / 3)
    env.set_state(MujocoRigidState(
        qpos=d["qpos"][pose_index],qvel=d["qvel"][pose_index],mocap_pos=d["mocap_pos"][pose_index],
        mocap_quat=d["mocap_quat"][pose_index],rod_yaw=float(d["rod_yaw"][pose_index]),
        step=int(d["step"][pose_index]),time=float(d["time"][pose_index]),
        integration_state=d["integration_state"][pose_index],integration_spec=int(d["integration_spec"])))
    colors={
        "board_geom":(.965,.949,.910,1),
        "obstacle_geom":(.16,.17,.16,1),
        "ball_geom":(.94,.51,.07,1),
        "rod_geom":(.10,.28,.72,1),
        "goal_geom":(.54,.69,.49,.55),
    }
    for name,color in colors.items():
        idx=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,name)
        env.model.geom_rgba[idx]=color
    env.model.light_diffuse[:] = .70
    env.model.vis.headlight.diffuse[:] = .20
    env.model.vis.headlight.ambient[:] = .20
    env.model.vis.headlight.specular[:] = .08
    env.model.vis.global_.offwidth=RESOLUTION
    env.model.vis.global_.offheight=RESOLUTION
    world=d["qpos"][:end+1,:3].copy()
    local=(R.T@world.T).T
    np.testing.assert_allclose(local[:,:2],d["observations"][:end+1,:2],atol=2e-7)
    with mujoco.Renderer(env.model,RESOLUTION,RESOLUTION,max_geom=1000) as renderer:
        renderer.update_scene(env.data,camera="orbit")
        s=renderer.scene
        def line(a,b,radius,color,kind=mujoco.mjtGeom.mjGEOM_CAPSULE):
            g=s.geoms[s.ngeom]
            mujoco.mjv_initGeom(g,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32))
            mujoco.mjv_connector(g,kind,radius,np.asarray(a),np.asarray(b))
            g.segid=s.ngeom
            g.objid=-1
            g.objtype=int(mujoco.mjtObj.mjOBJ_UNKNOWN)
            s.ngeom+=1
        for a,b in zip(world[:-1],world[1:]):
            line(a,b,.007,(.161,.616,.561,1))
        for t in [end//6,2*end//3]:
            line(world[t-2],world[t+2],.014,(.153,.278,.325,1),mujoco.mjtGeom.mjGEOM_ARROW)
        theta=np.linspace(0,2*np.pi,33)
        ring=[world[-1]+R@np.array([.026*np.cos(t),.026*np.sin(t),.003]) for t in theta]
        for a,b in zip(ring[:-1],ring[1:]):line(a,b,.0045,(.153,.278,.325,1))
        frame=renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation=renderer.render().copy()
        ground_id=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,"display_ground")
        floor=(segmentation[:,:,0]==ground_id) & (segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
    assert before==(path.stat().st_size,path.stat().st_mtime_ns)
    provenance={
        "panel": ["left","center","right"][case-3],
        "condition": case,"lateral_tilt_degrees":lateral,"longitudinal_tilt_degrees":20,
        "camera_offset_degrees":view_offset,
        "camera_convention":"positive = camera to the right of head-on; negative = to the left",
        "camera_position":position.tolist(),"camera_xyaxes":axes.tolist(),
        "camera_target":TARGET.tolist(),"camera_elevation_degrees":ELEVATION_DEGREES,
        "camera_distance":DISTANCE,"camera_fovy_degrees":42,
        "source":str(path.relative_to(ROOT)),"parent":meta["episode_id"],
        "controller":"recorded reference feedback controller; not a learned-policy result",
        "route_side":-1,"ordinal":0,"split":"train",
        "success":True,"success_step":end,"original_steps":len(d["actions"]),
        "final_speed_m_s":float(np.linalg.norm(d["observations"][end,2:4])),
        "terminal_board_xy":d["observations"][end,:2].tolist(),
        "rendered_state_index":pose_index,"rendered_time_seconds":float(d["time"][pose_index]),
        "pose_selection":"nearest recorded time index to one third of success-truncated trajectory","trajectory_points":len(world),"trajectory_smoothed":False,
        "rotation_board_to_world":R.tolist(),"board_gravity":(R.T@np.array([0,0,-9.81])).tolist(),
        "display_ground":"shared gray display floor; non-contact planes clipped to one common footprint",
        "new_model_queries":0,"dynamics_steps":0,"source_metadata_unchanged":True,
    }
    rows=[dict(panel=provenance["panel"],parent=meta["episode_id"],step=t,
               board_x=float(o[0]),board_y=float(o[1]),world_x=float(p[0]),world_y=float(p[1]),
               world_z=float(p[2]),terminal=t==end)
          for t,(o,p) in enumerate(zip(d["observations"][:end+1],world))]
    return frame,floor,provenance,rows

def main():
    SOURCE.mkdir(parents=True,exist_ok=True)
    frames=[];floors=[];records=[];points=[]
    for args in PANELS:
        frame,floor,meta,rows=render_panel(*args)
        frames.append(frame);floors.append(floor);records.append(meta);points+=rows
    # Shared crop/scale uses task geometry, excluding the infinite display floor.
    bboxes=[]
    for frame,floor in zip(frames,floors):
        mask=(~floor)&np.any(frame<245,axis=2)
        y,x=np.where(mask)
        bboxes.append((x.min(),y.min(),x.max()+1,y.max()+1))
    crop=(max(0,min(b[0] for b in bboxes)-90),max(0,min(b[1] for b in bboxes)-40),
          min(RESOLUTION,max(b[2] for b in bboxes)+90),min(RESOLUTION,max(b[3] for b in bboxes)+180))
    panels=[Image.fromarray(frame).crop(crop) for frame in frames]
    masks=[np.array(Image.fromarray(floor).crop(crop)) for floor in floors]
    w,h=panels[0].size
    margin=24
    total_w=w*3
    # A single ground footprint spans all views. White surrounds the floor and
    # boards. Segmentation preserves task geometry above the floor's far edge.
    footprint=Image.new("L",(total_w,h),0)
    ImageDraw.Draw(footprint).polygon([(100,int(h*.23)),(total_w-100,int(h*.23)),
                                      (total_w-12,h-24),(12,h-24)],fill=255)
    ground_mask=np.asarray(footprint)>0
    combined_array=np.full((h,total_w,3),255,dtype=np.uint8)
    for i,(panel,floor) in enumerate(zip(panels,masks)):
        rgb=np.array(panel)
        keep=(~floor)|ground_mask[:,i*w:(i+1)*w]
        rgb[~keep]=255
        combined_array[:,i*w:(i+1)*w]=rgb
        Image.fromarray(rgb).save(SOURCE/f"{records[i]['panel']}.png")
    combined=Image.new("RGB",(total_w+2*margin,h+2*margin),"white")
    combined.paste(Image.fromarray(combined_array),(margin,margin))
    combined.save(OUT/"environment.png",dpi=(450,450))
    apply_style()
    width=170/25.4
    fig=plt.figure(figsize=(width,width*combined.height/combined.width),facecolor="white")
    ax=fig.add_axes([0,0,1,1]);ax.set_facecolor("white");ax.imshow(combined);ax.axis("off")
    for extension in ["pdf","svg"]:
        fig.savefig(OUT/f"environment.{extension}",dpi=450,facecolor="white",pad_inches=0)
    plt.close(fig)
    assert not ET.parse(OUT/"environment.svg").findall(".//{http://www.w3.org/2000/svg}text")
    # Sky/background corners must be truly white in the exported figure.
    image=np.array(combined)
    assert (image[0,0]==255).all() and (image[-1,-1]==255).all()
    with (SOURCE/"measured_trajectories.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    (SOURCE/"provenance.json").write_text(json.dumps({
        "panels":records,"shared_crop":crop,"pixel_size":list(combined.size),
        "background":"white","text_in_figure":False,"scene_renders":3,
        "new_scientific_experiments":0,"source_metadata_checks":"size and modification time; no fingerprints",
    },indent=2,default=lambda x:int(x))+"\n")
    print(json.dumps({"figure":str(OUT/"environment.png"),"pixels":combined.size,
      "parents":[r["parent"] for r in records],"success_steps":[r["success_step"] for r in records],
      "camera_offsets":[r["camera_offset_degrees"] for r in records],"text_elements":0,"dynamics_steps":0}))
if __name__=="__main__":main()
