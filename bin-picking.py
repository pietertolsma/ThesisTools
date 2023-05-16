import os
from omni.isaac.kit import SimulationApp
import argparse
import cv2

parser = argparse.ArgumentParser(description="Bin Picking Generation Script")
parser.add_argument('amount', type=int, help='How many scenes to render')
parser.add_argument('--offset', action='store_true', help='whether to offset the cameras')
parser.add_argument('--material', choices=['diffuse', 'specular', 'clear', 'all'], help='type of material')
parser.add_argument('dataset_name', type=str, help='Output folder name')
args = parser.parse_args()

# Set rendering parameters and create an instance of kit
CONFIG = {
	"renderer": "PathTracing",
	# "renderer": "RayTracedLighting",
	"headless": True, 
	"width": 768, 
	"height": 768,
	"seg_width": 768,
	"seg_height": 768, 
	"num_frames": args.amount * 2
}

list_of_camera_positions = [
	(-20, -10, 100), 
	(0, -10, 100), 
	(20, -10, 100), 
	(-20, 10, 100), 
	(0, 10, 100), 
	(20, 10, 100)
]

list_of_camera_positions = list_of_camera_positions[:3] if not args.offset else list_of_camera_positions[3:]

SCOPE_NAME = "/MyScope"
tote_asset_path_path = '/home/fizyr/Documents'

assets_we_want_to_add_to_volume_path = '/home/fizyr/Documents/shapes/'


material = args.material

assets_we_want_to_add_to_volume = [
	"diff_cup-with-waves.usd", 
	"diff_glass-square-potion.usd", 
	"diff_square-plastic-bottle.usd",
	"diff_flower-bath-bomb.usd",
	"diff_heart-bath-bomb.usd",             
	"diff_star-bath-bomb.usd",
	"diff_glass-round-potion.usd", 
	"diff_seamless-plastic-champagne-glass.usd",
	"diff_tree-bath-bomb.usd",
]

number_of_objects = 8

kit = SimulationApp(launch_config=CONFIG)

import carb
import random
import math
import numpy as np
import time
from pxr import UsdGeom, Usd, Gf, UsdPhysics, PhysxSchema, Sdf
from PIL import Image
from pxr import UsdGeom
import json

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, lookat_to_quatf
from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid, get_all_matching_child_prims, set_prim_property, create_prim, delete_prim, get_prim_path
from pxr.UsdShade import Material, MaterialBindingAPI, Shader, Tokens, NodeGraph


import omni.replicator.core as rep
from omni.replicator.core import Writer, AnnotatorRegistry
from scipy.spatial.transform import Rotation as R
from omni.syntheticdata import helpers

random.seed(10)

def get_all_children_meshes(children):
	meshes = []
	for child in children:
		meshes.append(child)
	if child.GetChildren():
		meshes = meshes + get_all_children_meshes(child.GetChildren())
	return meshes


def get_internal_material_prims(children):
	materials = set()
	for child in children:
		if child.GetTypeName() == "Scope":
			materials = get_internal_material_prims(child.GetChildren())
		elif child.GetTypeName() == "Material":
			materials.add(child)
	return materials	

def generate_material_properties(dataset: str):
	types = ["clear", "diffuse", "specular"]
	assert dataset in types

	colors_1 = [[39/255, 60/255, 117/255], [232/255, 65/255, 24/255], [140/255, 122/255, 230/255], [251/255, 197/255, 49/255]]
	colors_2 = [[223/255, 249/255, 251/255], [246/255, 229/255, 151/255], [1.0, 1.0, 1.0]]

	res = {}

	res["glass_or_not"] = {
		"blend_weight": {
			"value" : 0.0 if dataset != "clear" else 1.0,
			"datatype": Sdf.ValueTypeNames.Float
		}
	}
	res["flake_or_plastic"] = {
		"blend_weight": {
			"value" : float(random.randint(0, 1)),
			"datatype": Sdf.ValueTypeNames.Float
		}
	}
	spec_rough = random.randint(0, 5) / 100
	res["roughness"] = {
		"f": {
			"value": [spec_rough, 1.0, 0.2][types.index(dataset)],
			"datatype": Sdf.ValueTypeNames.Float
		}
	}
	
	res["bump_noise"] = {
		"factor": {
			"value": random.choice([
				[0.0, 0.5, 1.0],
				[0.0, 1.0, 2.0],
				[0.0, 0.5, 1.0]
			][types.index(dataset)]),
			"datatype": Sdf.ValueTypeNames.Float
		}
	}
	
	res["checker_texture"] = {
		"scaling": {
			"value": tuple(random.choice([
				[0.0, 0.0, 0.0],
				[0.0, 4.0, 0.0],
				[4.0, 0.0, 0.0],
				[0.0, 30.0, 0.0],
				[30.0, 0.0, 0.0]
			])),
			"datatype": Sdf.ValueTypeNames.Float3
		}
	}
	res["color_1"] = {
		"c": {
			"value": tuple(random.choice(colors_1)),
			"datatype": Sdf.ValueTypeNames.Color3f
		}
	}
	res["color_2"] = {
		"c": {
			"value": tuple(random.choice(colors_2)),
			"datatype": Sdf.ValueTypeNames.Color3f
		}
	}
	return res
		

def randomize_object_color(base_prim_path, material_types):
	prim = get_prim_at_path(base_prim_path)

	xform_prims = prim.GetChildren()

	tote_shader_prim = get_prim_at_path("/Replicator/Ref_Xform/Ref/Looks/Plastic_Gray_A/Shader")

	tote_shader = Shader(tote_shader_prim)

	TOTE = [[1.0, 218/255, 121/255], [1.0, 82/255, 82/255], [132/255, 129/255, 122/255], [43/255, 50/255, 112/255]]
	tote_diffuse_tint_value = random.choice(list(TOTE))
	tote_diffuse_tint = tote_shader.CreateInput(f"diffuse_tint", Sdf.ValueTypeNames.Color3f)
	tote_diffuse_tint.Set(tuple(tote_diffuse_tint_value))

	# base_asset_path = '/home/fizyr/Documents/texture-normal-maps/'
	# detail_normal_map_files = os.listdir(base_asset_path)

	material_type_index = 0

	for xform_prim in xform_prims:
		# print(xform_prim)
		children = xform_prim.GetChildren()
		# print(children)

		meshes = [child for child in children if child.GetTypeName() == "Mesh"]

		parent_mesh = meshes[0]

		children_meshes = []
		if parent_mesh.GetChildren():
			children_meshes = get_all_children_meshes(parent_mesh.GetChildren())

		internal_material_prims = get_internal_material_prims(children)

		if not internal_material_prims:
			print("The object does not have any internal materials")

		material_prims = list(internal_material_prims)

		material = material_types[material_type_index]
		material_type_index += 1
		material_type_index = 0 if material_type_index >= len(material_types) else material_type_index
		# TODO: Check if this works
		props = generate_material_properties(material)

		for material_prim in material_prims:
			material_path = material_prim.GetPath()
			material_prim = get_prim_at_path(material_path)
			prim_paths = material_prim.GetChildren()

			for prim_path in prim_paths:
				if prim_path.GetTypeName() == 'Shader':
					#print(prim_path.prim_path)
					shader_object = Shader(prim_path)
				else:
					shader_object = NodeGraph(prim_path)

				shader_name = str(prim_path).split("/")[-1]	[:-2]
				for field in props.get(shader_name, {}).keys():
					val = props[shader_name][field]["value"]
					dtype = props[shader_name][field]["datatype"]

					prop = shader_object.CreateInput(field, dtype)
					prop.Set(val)

def delete_sphere_light(): 
	delete_prim('/Replicator/SphereLight_Xform')

def simulate_falling_objects(index, num_sim_steps=2000):
	world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)
	world.scene.add_ground_plane(size=200)

	if index > 0:
		for i in range(number_of_objects):
			prim_name = f"item_{i}"
			world.scene._scene_registry.remove_object(prim_name)
		
		remove_falling_objects()

	scene_metadata = {}

	# Create a simulation ready world
	objects_to_place_in_scene = []
	for i in range(number_of_objects):
		selected_object = random.choice(list(assets_we_want_to_add_to_volume))
		objects_to_place_in_scene.append(selected_object)
		scene_metadata[f"/MyScope/item_{i}"] = {
			"model": selected_object
		}


	# Spawn boxes falling on the pallet
	for i, object_name in enumerate(objects_to_place_in_scene):
			# Spawn box prim
			prim_name = f"item_{i}"
			item_prim = prims.create_prim(
					prim_path=f"{SCOPE_NAME}/{prim_name}",
					usd_path=assets_we_want_to_add_to_volume_path + object_name,
					semantic_label=f"item",
			)

			# Wrap the cardbox prim into a rigid prim to be able to simulate it
			rigid_prim = RigidPrim(
				prim_path=str(item_prim.GetPrimPath()),
				name=prim_name,
				position=(0,0,20),
			)

			# Make sure physics are enabled on the rigid prim
			rigid_prim.enable_rigid_body_physics()

			world.scene.add(rigid_prim)

	# Pass material type to randomize object color
	material_types = []
	for i, object_name in enumerate(objects_to_place_in_scene):
		if args.material == 'all':
			material = random.choice(['diffuse', 'clear', 'specular'])
		else:
			material = args.material
		material_types.append(material)
		scene_metadata[f"/MyScope/item_{i}"]["material"] = material

	randomize_object_color("/MyScope", material_types)
	world.reset()

	# Reset world after adding simulated assets for physics handles to be propagated properly

	# Simulate the world for the given number of steps or until the highest box stops moving
	last_item = world.scene.get_object(f"item_{number_of_objects - 1}")
	for i in range(num_sim_steps):
			world.step(render=False)
			if last_item and np.linalg.norm(last_item.get_linear_velocity()) < 0.001 and i > 500:
					print(f"Simulation stopped after {i} steps")
					break
	return scene_metadata

def sphere_lights(num):
	array_of_available_positions = [(0, 0, 200), (100, 100, 200), (-100, -100, 200)]

	selected_position = random.choice(list(array_of_available_positions))

	lights = rep.create.light(
		light_type="Sphere",
		temperature=5500,
		intensity=11000,
		position=selected_position,	
		scale=200,
		count=num
	)
	return lights.node

	# print(selected_position)

rep.randomizer.register(sphere_lights)


def dome_light():
	dome_light = rep.create.light(
		# rotation=rep.distribution.uniform((-180,-180,-180), (180,180,180)),
		texture='/home/fizyr/Downloads/TexturesCom_HDRPanorama183_header.jpg',
		light_type="dome"
	)
	return dome_light.node

rep.randomizer.register(dome_light)



def createCameras(list_of_camera_positions): 
	list_of_cameras = []
	for camera_position in list_of_camera_positions:
		camera = rep.create.camera(position=camera_position, clipping_range=(0.01, 10000.0), focus_distance=45.0, look_at=(0,0,0))
		list_of_cameras.append(camera)
	return list_of_cameras


def createRenderProducts(list_of_cameras): 
	list_of_render_products = []
	for camera in list_of_cameras:
		render_product = rep.create.render_product(camera, (CONFIG["width"], CONFIG["height"]))
		list_of_render_products.append(render_product)
	return list_of_render_products


def remove_falling_objects():
	for i in range(number_of_objects):
		prim_name = f"item_{i}"
		prim=get_prim_at_path(f"{SCOPE_NAME}/{prim_name}")
		prim_path = get_prim_path(prim)
		delete_prim(prim_path)


def save_rgb(rgb_data, depth_data, file_name):
	#dat = np.concatenate((rgb_data, depth_data[:, :, None]), axis=2)
	#print(dat.shape)

	# normalize the depth data
	dmin, dmax = depth_data.min(), depth_data.max()
	dnorm = 255 * ((depth_data - dmin) / (dmax - dmin))

	file_name = file_name + f"_{int(dmin)}_{int(dmax)}"

	rgb_data[:, :, 3] = dnorm
	rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
	#print(rgb_data.shape)
	#print(depth_data.shape)
	#rgb_image_data[:, :, 3] = np.frombuffer(depth_data, dtype=np.uint8).reshape(*depth_data.shape, -1)
	rgb_img = Image.fromarray(rgb_image_data, "RGBA")
	rgb_img.save(file_name + ".png")


def save_distance(distance_data, file_name):
	# print(f"{distance_data.min()} {distance_data.max()}")
	np.save(file_name, distance_data.astype(np.uint8))
	# distance_data = distance_data.tolist()

	# with open(file_name, "w") as outfile:
		# json.dump(distance_data, outfile)


def save_instance(instance_data, file_name):
	data = np.array(instance_data['data'], dtype=np.uint8)

	resized = cv2.resize(data, (CONFIG["seg_height"], CONFIG["seg_width"]), interpolation=cv2.INTER_NEAREST)
	#resized = resized.astype(np.uint8)

	np.save(file_name, resized)

	# instance_data['data'] = instance_data['data'].tolist()
	# with open(file_name, "w") as outfile:
	# 	json.dump(instance_data, outfile)


def writeCameraSpecificationsToJsonFile():
	with open('data.json', 'w') as f:
		json.dump(list_of_camera_matrixes, f, indent=4)


def createCameraSpecificaton(rotation, camera_position):
		test = {'rx': rotation[0], 'ry': rotation[1], 'rz': rotation[2], 'camera_position': camera_position}
		list_of_camera_matrixes.append(test)


def write_camera_intrinsics(list_of_camera_intrinsics):
	file_path = os.path.join(os.getcwd(), args.dataset_name, "")
	dir = os.path.dirname(file_path)
	file_path = dir + "/cameras"

	out = []
	if os.path.exists(file_path):
		out = json.load(open(file_path))
	existing_positions = [item['position'] for item in out]
	#print(existing_positions)
	for i, item in enumerate(list_of_camera_intrinsics):
		#print(item['position'])
		if list(item['position']) in existing_positions:
			continue
		out.append(item)

	with open(file_path, "w") as outfile:
		json.dump(out, outfile)


def get_camera_intrinsics(index, child_of_child, list_of_camera_intrinsics, list_of_camera_positions, list_of_camera_rotations):
	focal_length = child_of_child.GetAttribute("focalLength").Get()
	horiz_aperture = child_of_child.GetAttribute("horizontalAperture").Get()
	vert_aperture = child_of_child.GetAttribute("verticalAperture").Get()
	# Pixels are square so we can also do:
	near, far = child_of_child.GetAttribute("clippingRange").Get()
	fov = 2 * math.atan(horiz_aperture / (2 * focal_length))

	# compute focal point and center
	focal_x = CONFIG["height"] * focal_length / vert_aperture
	focal_y = CONFIG["width"] * focal_length / horiz_aperture
	center_x = CONFIG["height"] * 0.5
	center_y = CONFIG["width"] * 0.5

	camera_intrinsics = {
		"position": list_of_camera_positions[index],
		"rotation": list_of_camera_rotations[index],
		"focal_length": focal_length,
		"horiz_aperture": horiz_aperture,
		"vert_aperture": vert_aperture,

		"near": near,
		"far": far,
		"fov": fov,

		"focal_x": focal_x,
		"focal_y": focal_y,
		"center_x": center_x,
		"center_y": center_y,
	}

	list_of_camera_intrinsics.append(camera_intrinsics)
	return list_of_camera_intrinsics


class MyWriter(Writer):
	def __init__(self, rgb: bool = True, distance_to_camera: bool = True, instance_segmentation: bool = True):
		self._frame_id = 0

		if rgb:
			self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
		if distance_to_camera:
			self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_image_plane"))
		if instance_segmentation:
			self.annotators.append(AnnotatorRegistry.get_annotator("instance_segmentation"))

	def write(self, data):
		self._frame_id += 1


def rotate_to_center(camera_pos: np.array):
	min_z = np.array([0, 0, -1])
	center_dir = -camera_pos / np.linalg.norm(camera_pos)
	sign = -1 * np.sign(center_dir[0])
	y_angle = sign * np.arccos(np.dot(min_z, center_dir))
	y_rot = R.from_euler("XYZ", angles=[0, y_angle, 0], degrees=False).as_matrix()
	cur_orientation = y_rot @ min_z
	sign = -1 * np.sign(cur_orientation[1])
	dot = np.dot(cur_orientation, center_dir)
	z_angle = 0 if dot == 0 else sign * np.arccos(dot)
	return 0, 180 * (y_angle / np.pi), 180 * (z_angle / np.pi)

def main():
	rep.WriterRegistry.register(MyWriter)

	# Create a custom scope for newly added prims
	stage = get_current_stage()
	scope = UsdGeom.Scope.Define(stage, SCOPE_NAME)		

	# Create annotator output directory
	file_path = os.path.join(os.getcwd(), args.dataset_name, "")
	print(f"Writing annotator data to {file_path}")
	dir = os.path.dirname(file_path)
	os.makedirs(dir, exist_ok=True)

	# simulate_falling_objects()
	with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):
		# rep.randomizer.sphere_lights(1)
		rep.randomizer.dome_light()

	tote_asset = rep.create.from_dir(tote_asset_path_path, path_filter="Tote_A04_80x60x22cm_PR_V_NVD_01.usd")
	# randomize_object_color("/Replicator/Ref_Xform")

	list_of_cameras = []

	list_of_camera_intrinsics = []
	list_of_camera_rotations = []
	for camera_position in list_of_camera_positions:
		rotation = rotate_to_center(np.array(camera_position))
		isaac_rotation = (rotation[0] -90, rotation[1] -90, rotation[2])
		camera = rep.create.camera(position=camera_position, clipping_range=(0.01, 10000.0), focus_distance=100.0, rotation=(isaac_rotation))
		list_of_cameras.append(camera)
		list_of_camera_rotations.append(rotation)

	replicator = get_prim_at_path("/Replicator")
	children = replicator.GetChildren()
	index = 0
	for child in children:

		child_of_children = child.GetChildren()
		for child_of_child in child_of_children:

			if child_of_child.GetTypeName() == "Camera":
				list_of_camera_intrinsics = get_camera_intrinsics(index, child_of_child, list_of_camera_intrinsics, list_of_camera_positions, list_of_camera_rotations)
				index += 1

	write_camera_intrinsics(list_of_camera_intrinsics)

	list_of_render_products = []

	for camera in list_of_cameras:
			render_product = rep.create.render_product(camera, (CONFIG["width"], CONFIG["height"]))
			list_of_render_products.append(render_product)

	for i, val in enumerate(list_of_render_products):
			globals()["render_product" + str(i+1)] = val


	# Acess the data through a custom writer
	writer = rep.WriterRegistry.get("MyWriter")
	writer.initialize(rgb=True, distance_to_camera =True, instance_segmentation = True)
	writer.attach([render_product1, render_product2, render_product3])


	# Acess the data through annotators
	rgb_annotators = []
	distance_to_camera_annotators = []
	instance_segmentation_annotators = []

	for rp in [render_product1, render_product2, render_product3]:

			rgb = rep.AnnotatorRegistry.get_annotator("rgb")
			rgb.attach([rp])
			rgb_annotators.append(rgb)

			distance_to_camera = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
			distance_to_camera.attach([rp])
			distance_to_camera_annotators.append(distance_to_camera)

			instance_segmentation = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
			instance_segmentation.attach([rp])
			instance_segmentation_annotators.append(instance_segmentation)

	for i in range(CONFIG["num_frames"] // 2):
		#TODO fix moving lights
		with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):
			rep.randomizer.sphere_lights(1)
			# rep.randomizer.dome_light()

		kit.reset_render_settings()

		scene_metadata = simulate_falling_objects(i)

		#randomize_object_color("/Replicator/Ref_Xform")
	
		rep.orchestrator.step()

		offset = 3 if args.offset else 0

		#TODO combine rgb and dept data into one png
		for j, rgb_annot in enumerate(rgb_annotators):
			save_rgb(rgb_annot.get_data(), distance_to_camera_annotators[j].get_data(), f"{dir}/rp{j + offset}_step_{i}")

		# for k, distance_to_camera_annot in enumerate(distance_to_camera_annotators):
		# 	save_distance(distance_to_camera_annot.get_data(), f"{dir}/distance_rp{k + offset}_step_{i}")
		
		scene_metadata["mappings"] = instance_segmentation_annotators[0].get_data()["info"]
		json.dump(scene_metadata, open(f"{dir}/instance_step_{i}.json", 'w'))
		for l, instance_segmentation_annot in enumerate(instance_segmentation_annotators):
			save_instance(instance_segmentation_annot.get_data(), f"{dir}/instance_rp{l + offset}_step_{i}.npy")

		delete_sphere_light()

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		carb.log_error(f"Exception: {e}")
		import traceback

		traceback.print_exc()
	finally:
		kit.close()