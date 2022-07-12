# ------------------------------------------------------------------------------------------
# -                                 Modified Code From                                     -
# - https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/vis_gui.py -
# ------------------------------------------------------------------------------------------

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import cc3d
import pandas as pd

from utils import *

isMacOS = (platform.system() == "Darwin")

class Settings:
	LIT = "defaultLit"
	UNLIT = "defaultUnlit"
	NORMALS = "normals"
	DEPTH = "depth"

	DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
	POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
	CUSTOM_PROFILE_NAME = "Custom"
	LIGHTING_PROFILES = {
		DEFAULT_PROFILE_NAME: {
			"ibl_intensity": 45000,
			"sun_intensity": 45000,
			"sun_dir": [0.577, -0.577, -0.577],
			# "ibl_rotation":
			"use_ibl": True,
			"use_sun": True,
		},
		"Bright day with sun at -Y": {
			"ibl_intensity": 45000,
			"sun_intensity": 45000,
			"sun_dir": [0.577, 0.577, 0.577],
			# "ibl_rotation":
			"use_ibl": True,
			"use_sun": True,
		},
		"Bright day with sun at +Z": {
			"ibl_intensity": 45000,
			"sun_intensity": 45000,
			"sun_dir": [0.577, 0.577, -0.577],
			# "ibl_rotation":
			"use_ibl": True,
			"use_sun": True,
		},
		"Less Bright day with sun at +Y": {
			"ibl_intensity": 35000,
			"sun_intensity": 50000,
			"sun_dir": [0.577, -0.577, -0.577],
			# "ibl_rotation":
			"use_ibl": True,
			"use_sun": True,
		},
		"Less Bright day with sun at -Y": {
			"ibl_intensity": 35000,
			"sun_intensity": 50000,
			"sun_dir": [0.577, 0.577, 0.577],
			# "ibl_rotation":
			"use_ibl": True,
			"use_sun": True,
		},
		"Less Bright day with sun at +Z": {
			"ibl_intensity": 35000,
			"sun_intensity": 50000,
			"sun_dir": [0.577, 0.577, -0.577],
			# "ibl_rotation":
			"use_ibl": True,
			"use_sun": True,
		},
		POINT_CLOUD_PROFILE_NAME: {
			"ibl_intensity": 60000,
			"sun_intensity": 50000,
			"use_ibl": True,
			"use_sun": False,
			# "ibl_rotation":
		},
	}

	DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
	PREFAB = {
		DEFAULT_MATERIAL_NAME: {
			"metallic": 0.0,
			"roughness": 0.7,
			"reflectance": 0.5,
			"clearcoat": 0.2,
			"clearcoat_roughness": 0.2,
			"anisotropy": 0.0
		},
		"Metal (rougher)": {
			"metallic": 1.0,
			"roughness": 0.5,
			"reflectance": 0.9,
			"clearcoat": 0.0,
			"clearcoat_roughness": 0.0,
			"anisotropy": 0.0
		},
		"Metal (smoother)": {
			"metallic": 1.0,
			"roughness": 0.3,
			"reflectance": 0.9,
			"clearcoat": 0.0,
			"clearcoat_roughness": 0.0,
			"anisotropy": 0.0
		},
		"Plastic": {
			"metallic": 0.0,
			"roughness": 0.5,
			"reflectance": 0.5,
			"clearcoat": 0.5,
			"clearcoat_roughness": 0.2,
			"anisotropy": 0.0
		},
		"Glazed ceramic": {
			"metallic": 0.0,
			"roughness": 0.5,
			"reflectance": 0.9,
			"clearcoat": 1.0,
			"clearcoat_roughness": 0.1,
			"anisotropy": 0.0
		},
		"Clay": {
			"metallic": 0.0,
			"roughness": 1.0,
			"reflectance": 0.5,
			"clearcoat": 0.1,
			"clearcoat_roughness": 0.287,
			"anisotropy": 0.0
		},
	}

	def __init__(self):
		self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
		self.bg_color = gui.Color(1, 1, 1)
		self.show_skybox = False
		self.show_axes = True
		self.use_ibl = True
		self.use_sun = True
		self.new_ibl_name = None  # clear to None after loading
		self.ibl_intensity = 45000
		self.sun_intensity = 45000
		self.sun_dir = [0.577, -0.577, -0.577]
		self.sun_color = gui.Color(1, 1, 1)

		self.apply_material = True  # clear to False after processing
		self._materials = {
			Settings.LIT: rendering.MaterialRecord(),
			Settings.UNLIT: rendering.MaterialRecord(),
			Settings.NORMALS: rendering.MaterialRecord(),
			Settings.DEPTH: rendering.MaterialRecord()
		}
		self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
		self._materials[Settings.LIT].shader = Settings.LIT
		self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
		self._materials[Settings.UNLIT].shader = Settings.UNLIT
		self._materials[Settings.NORMALS].shader = Settings.NORMALS
		self._materials[Settings.DEPTH].shader = Settings.DEPTH

		# Conveniently, assigning from self._materials[...] assigns a reference,
		# not a copy, so if we change the property of a material, then switch
		# to another one, then come back, the old setting will still be there.
		self.material = self._materials[Settings.LIT]

	def set_material(self, name):
		self.material = self._materials[name]
		self.apply_material = True

	def apply_material_prefab(self, name):
		assert (self.material.shader == Settings.LIT)
		prefab = Settings.PREFAB[name]
		for key, val in prefab.items():
			setattr(self.material, "base_" + key, val)

	def apply_lighting_profile(self, name):
		profile = Settings.LIGHTING_PROFILES[name]
		for key, val in profile.items():
			setattr(self, key, val)


class AppWindow:
	MENU_OPEN = 1
	MENU_EXPORT = 2
	MENU_QUIT = 3
	MENU_SHOW_SETTINGS = 11
	MENU_ABOUT = 21

	DEFAULT_IBL = "default"

	MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
	MATERIAL_SHADERS = [
		Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
	]

	def __init__(self, width, height):
		self.settings = Settings()
		resource_path = gui.Application.instance.resource_path
		self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

		self.window = gui.Application.instance.create_window(
			"Open3D", width, height)
		w = self.window  # to make the code more concise

		# 3D widget
		self._scene = gui.SceneWidget()
		self._scene.scene = rendering.Open3DScene(w.renderer)
		# self._scene.set_on_sun_direction_changed(self._on_sun_dir)

		# Model name and filename lists

		self.modelFilenameList = []
		self.modelNameList = []
		self.origionalGeometries = {}

		# 2D Cloud Init
		self.tiffFile = ''
		self.tiffShape = [2, 2, 2]

		#Opacity
		self.opacity = 100.0

		# ---- Settings panel ----
		# Rather than specifying sizes in pixels, which may vary in size based
		# on the monitor, especially on macOS which has 220 dpi monitors, use
		# the em-size. This way sizings will be proportional to the font size,
		# which will create a more visually consistent size across platforms.
		em = w.theme.font_size
		separation_height = int(round(0.5 * em))

		# Widgets are laid out in layouts: gui.Horiz, gui.Vert,
		# gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
		# achieve complex designs. Usually we use a vertical layout as the
		# topmost widget, since widgets tend to be organized from top to bottom.
		# Within that, we usually have a series of horizontal layouts for each
		# row. All layouts take a spacing parameter, which is the spacing
		# between items in the widget, and a margins parameter, which specifies
		# the spacing of the left, top, right, bottom margins. (This acts like
		# the 'padding' property in CSS.)
		self._settings_panel = gui.Vert(
			0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

		# Create a collapsible vertical widget, which takes up enough vertical
		# space for all its children when open, but only enough for text when
		# closed. This is useful for property pages, so the user can hide sets
		# of properties they rarely use.
		view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
										 gui.Margins(em, 0, 0, 0))
		view_ctrls.set_is_open(False)

		self._arcball_button = gui.Button("Arcball")
		self._arcball_button.horizontal_padding_em = 0.5
		self._arcball_button.vertical_padding_em = 0
		self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
		self._fly_button = gui.Button("Fly")
		self._fly_button.horizontal_padding_em = 0.5
		self._fly_button.vertical_padding_em = 0
		self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
		self._model_button = gui.Button("Model")
		self._model_button.horizontal_padding_em = 0.5
		self._model_button.vertical_padding_em = 0
		self._model_button.set_on_clicked(self._set_mouse_mode_model)

		view_ctrls.add_child(gui.Label("Mouse controls"))
		# We want two rows of buttons, so make two horizontal layouts. We also
		# want the buttons centered, which we can do be putting a stretch item
		# as the first and last item. Stretch items take up as much space as
		# possible, and since there are two, they will each take half the extra
		# space, thus centering the buttons.
		h = gui.Horiz(0.25 * em)  # row 1
		h.add_stretch()
		h.add_child(self._arcball_button)
		h.add_child(self._fly_button)
		h.add_child(self._model_button)
		h.add_stretch()
		view_ctrls.add_child(h)
		view_ctrls.add_fixed(separation_height)

		self._bg_color = gui.ColorEdit()
		self._bg_color.set_on_value_changed(self._on_bg_color)

		grid = gui.VGrid(2, 0.25 * em)
		grid.add_child(gui.Label("BG Color"))
		grid.add_child(self._bg_color)
		view_ctrls.add_child(grid)

		self._show_axes = gui.Checkbox("Show axes")
		self._show_axes.set_on_checked(self._on_show_axes)
		view_ctrls.add_fixed(separation_height)
		view_ctrls.add_child(self._show_axes)

		view_ctrls.add_fixed(separation_height)
		self._settings_panel.add_child(view_ctrls)



		########################################################################################



		material_settings = gui.CollapsableVert("Material settings", 0,
												gui.Margins(em, 0, 0, 0))
		material_settings.set_is_open(False)

		self._shader = gui.Combobox()
		self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
		self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
		self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
		self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
		self._shader.set_on_selection_changed(self._on_shader)
		self._material_prefab = gui.Combobox()
		for prefab_name in sorted(Settings.PREFAB.keys()):
			self._material_prefab.add_item(prefab_name)
		self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
		self._material_prefab.set_on_selection_changed(self._on_material_prefab)
		self._material_color = gui.ColorEdit()
		self._material_color.set_on_value_changed(self._on_material_color)
		self._point_size = gui.Slider(gui.Slider.INT)
		self._point_size.set_limits(1, 10)
		self._point_size.set_on_value_changed(self._on_point_size)

		self._opacity_slider = gui.Slider(gui.Slider.INT)
		self._opacity_slider.set_limits(1, 100)
		self._opacity_slider.set_on_value_changed(self._on_opacity)

		grid = gui.VGrid(2, 0.25 * em)
		grid.add_child(gui.Label("Type"))
		grid.add_child(self._shader)
		grid.add_child(gui.Label("Material"))
		grid.add_child(self._material_prefab)
		grid.add_child(gui.Label("Color"))
		grid.add_child(self._material_color)
		grid.add_child(gui.Label("Point size"))
		grid.add_child(self._point_size)
		grid.add_child(gui.Label("Opacity"))
		grid.add_child(self._opacity_slider)
		material_settings.add_child(grid)

		material_settings.add_fixed(separation_height)
		self._settings_panel.add_child(material_settings)
		# ----

		# Normally our user interface can be children of all one layout (usually
		# a vertical layout), which is then the only child of the window. In our
		# case we want the scene to take up all the space and the settings panel
		# to go above it. We can do this custom layout by providing an on_layout
		# callback. The on_layout callback should set the frame
		# (position + size) of every child correctly. After the callback is
		# done the window will layout the grandchildren.
		w.set_on_layout(self._on_layout)
		w.add_child(self._scene)
		w.add_child(self._settings_panel)


		###########################################################################################
		# Add Remove Models
		models = gui.CollapsableVert("Model Manipulation", 0,
										gui.Margins(em, 0, 0, 0))
		models.set_is_open(False)

		self._model_to_remove_combo = gui.Combobox()
		self._model_to_remove_combo.add_item('No Models Yet')

		h = gui.Horiz(0.25 * em)  # row 1
		h.add_stretch()
		h.add_child(gui.Label("Model To Edit:"))
		h.add_child(self._model_to_remove_combo)
		models.add_child(h)
		models.add_fixed(separation_height)

		showHideButton = gui.Button('Show / Hide')
		showHideButton.set_on_clicked(self.showHideButtonClick)
		models.add_child(showHideButton)
		models.add_fixed(separation_height)

		removeButton = gui.Button("Remove")
		removeButton.set_on_clicked(self.removeButtonClick)
		models.add_child(removeButton)
		models.add_fixed(separation_height)

		models.add_child(gui.Label("Paint Model a Solid Color"))
		models.add_fixed(separation_height)

		self._paint_color = gui.ColorEdit()
		models.add_child(self._paint_color)
		models.add_fixed(separation_height)

		self.paint_button = gui.Button("Paint Uniform Color")
		self.paint_button.set_on_clicked(self.paint_geometry)
		models.add_child(self.paint_button)
		models.add_fixed(separation_height)

		models.add_child(gui.Label("Crop Models"))
		models.add_fixed(separation_height)

		h = gui.Horiz(0.25 * em)
		h.add_stretch()
		h.add_child(gui.Label("X, Y, and Z Min:"))
		self.cropMinEdit = gui.VectorEdit()
		h.add_child(self.cropMinEdit)
		models.add_child(h)

		models.add_fixed(separation_height)

		h = gui.Horiz(0.25 * em)
		h.add_stretch()
		h.add_child(gui.Label("X, Y, and Z Max:"))
		self.cropMaxEdit = gui.VectorEdit()
		h.add_child(self.cropMaxEdit)
		models.add_child(h)

		models.add_fixed(separation_height)
		cropButton = gui.Button('Crop')
		cropButton.set_on_clicked(self.cropButtonClick)
		models.add_child(cropButton)

		models.add_fixed(separation_height)
		uncropButton = gui.Button('Uncrop / Restore')
		uncropButton.set_on_clicked(self.uncropButtonClick)
		models.add_child(uncropButton)

		models.add_fixed(separation_height)
		getStatsButton = gui.Button("Get Statistics for Model (Semantic)")
		getStatsButton.set_on_clicked(self.getStatsButtonClickedSemantic)
		models.add_child(getStatsButton)

		models.add_fixed(separation_height)
		self._settings_panel.add_child(models)

		###########################################################################
		# 2D Image Showing

		twoDImage = gui.CollapsableVert("2D Image", 0,
								gui.Margins(em, 0, 0, 0))
		twoDImage.set_is_open(False)

		twoDImage.add_child(gui.Label("Add a slice of a 2D image stack\ninto the scene"))

		pickTiffButton = gui.Button('Pick Image')
		pickTiffButton.set_on_clicked(self._on_menu_open_tiff)
		twoDImage.add_child(pickTiffButton)
		twoDImage.add_fixed(separation_height)

		h = gui.Horiz(0.25 * em)
		#h.add_stretch()
		h.add_child(gui.Label("Choose axis for Slice:"))
		xButton = gui.Button('X')
		xButton.set_on_clicked(self.tiffXButtonClicked)
		h.add_child(xButton)
		yButton = gui.Button('Y')
		yButton.set_on_clicked(self.tiffYButtonClicked)
		h.add_child(yButton)
		zButton = gui.Button('Z')
		zButton.set_on_clicked(self.tiffZButtonClicked)
		h.add_child(zButton)
		twoDImage.add_child(h)

		twoDImage.add_child(gui.Label("Change Location of Slice:"))

		self.tiffAxis = 'x'
		self.tiffShape = [10, 10, 10]
		self.tiffSliderValue = 1
		self._sliderTiffLocation = gui.Slider(gui.Slider.INT)
		self._sliderTiffLocation.set_limits(0, 9)
		self._sliderTiffLocation.set_on_value_changed(self.tiffSliderChange)
		twoDImage.add_child(self._sliderTiffLocation)
		twoDImage.add_fixed(separation_height)

		tiffUpdateButton = gui.Button('Update Slice Location')
		tiffUpdateButton.set_on_clicked(self.tiffUpdate)
		twoDImage.add_child(tiffUpdateButton)
		twoDImage.add_fixed(separation_height)

		tiffRemoveButton = gui.Button('Remove 2D Slice')
		tiffRemoveButton.set_on_clicked(self.tiffRemove)
		twoDImage.add_child(tiffRemoveButton)
		twoDImage.add_fixed(separation_height)

		self._settings_panel.add_child(twoDImage)


		# ---- Menu ----
		# The menu is global (because the macOS menu is global), so only create
		# it once, no matter how many windows are created
		if gui.Application.instance.menubar is None:
			if isMacOS:
				app_menu = gui.Menu()
				app_menu.add_item("About", AppWindow.MENU_ABOUT)
				app_menu.add_separator()
				app_menu.add_item("Quit", AppWindow.MENU_QUIT)
			file_menu = gui.Menu()
			file_menu.add_item("Open...", AppWindow.MENU_OPEN)
			file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
			if not isMacOS:
				file_menu.add_separator()
				file_menu.add_item("Quit", AppWindow.MENU_QUIT)
			settings_menu = gui.Menu()
			settings_menu.add_item("Show / Hide Sidebar",
								   AppWindow.MENU_SHOW_SETTINGS)
			settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
			help_menu = gui.Menu()
			help_menu.add_item("About", AppWindow.MENU_ABOUT)

			menu = gui.Menu()
			if isMacOS:
				# macOS will name the first menu item for the running application
				# (in our case, probably "Python"), regardless of what we call
				# it. This is the application menu, and it is where the
				# About..., Preferences..., and Quit menu items typically go.
				menu.add_menu("Example", app_menu)
				menu.add_menu("File", file_menu)
				menu.add_menu("Settings", settings_menu)
				# Don't include help menu unless it has something more than
				# About...
			else:
				menu.add_menu("File", file_menu)
				menu.add_menu("Settings", settings_menu)
				menu.add_menu("Help", help_menu)
			gui.Application.instance.menubar = menu

		# The menubar is global, but we need to connect the menu items to the
		# window, so that the window can call the appropriate function when the
		# menu item is activated.
		w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
		w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
									 self._on_menu_export)
		w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
		w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
									 self._on_menu_toggle_settings_panel)
		w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
		# ----

		self._apply_settings()

	def paint_geometry(self):
		toPaint = self._model_to_remove_combo.selected_text
		colorToPaint = self._paint_color.color_value
		colorToPaint = colorToPaint.red, colorToPaint.green, colorToPaint.blue

		geometry_to_paint = self.cropButtonClick()
		geometry_to_paint.paint_uniform_color(colorToPaint)

		self._scene.scene.remove_geometry(toPaint)
		self._scene.scene.add_geometry(toPaint, geometry_to_paint, self.settings.material)

	def _apply_settings(self):

		currentlySelected = self._model_to_remove_combo.selected_text

		self._model_to_remove_combo.clear_items()
		if len(self.modelNameList) == 0:
			self._model_to_remove_combo.add_item('No Models Yet')

		else:
			for name in self.modelNameList:
				self._model_to_remove_combo.add_item(name)
		try:
			self._model_to_remove_combo.selected_text = currentlySelected
		except:
			pass

		bg_color = [
			self.settings.bg_color.red, self.settings.bg_color.green,
			self.settings.bg_color.blue, self.settings.bg_color.alpha
		]
		self._scene.scene.set_background(bg_color)
		self._scene.scene.show_skybox(self.settings.show_skybox)
		self._scene.scene.show_axes(self.settings.show_axes)
		if self.settings.new_ibl_name is not None:
			self._scene.scene.scene.set_indirect_light(
				self.settings.new_ibl_name)
			# Clear new_ibl_name, so we don't keep reloading this image every
			# time the settings are applied.
			self.settings.new_ibl_name = None
		self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
		self._scene.scene.scene.set_indirect_light_intensity(
			self.settings.ibl_intensity)
		sun_color = [
			self.settings.sun_color.red, self.settings.sun_color.green,
			self.settings.sun_color.blue
		]
		self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
											  self.settings.sun_intensity)
		self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

		if self.settings.apply_material:
			self.settings.material.base_color = (self.settings.material.base_color[0],self.settings.material.base_color[1],self.settings.material.base_color[2],float(self.opacity) / 100.0)
			self._scene.scene.update_material(self.settings.material)
			self.settings.apply_material = False

		self._bg_color.color_value = self.settings.bg_color
		self._show_axes.checked = self.settings.show_axes
		self._material_prefab.enabled = (
			self.settings.material.shader == Settings.LIT)
		c = gui.Color(self.settings.material.base_color[0],
					  self.settings.material.base_color[1],
					  self.settings.material.base_color[2],
					  self.settings.material.base_color[3])
		self._material_color.color_value = c
		self._point_size.double_value = self.settings.material.point_size

	def _on_layout(self, layout_context):
		# The on_layout callback should set the frame (position + size) of every
		# child correctly. After the callback is done the window will layout
		# the grandchildren.
		r = self.window.content_rect
		self._scene.frame = r
		width = 17 * layout_context.theme.font_size
		height = min(
			r.height,
			self._settings_panel.calc_preferred_size(
				layout_context, gui.Widget.Constraints()).height)
		self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
											  height)

	def _set_mouse_mode_rotate(self):
		self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

	def _set_mouse_mode_fly(self):
		self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

	def _set_mouse_mode_model(self):
		self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

	def _on_bg_color(self, new_color):
		self.settings.bg_color = new_color
		self._apply_settings()

	def _on_show_axes(self, show):
		self.settings.show_axes = show
		self._apply_settings()

	def _on_lighting_profile(self, name, index):
		if name != Settings.CUSTOM_PROFILE_NAME:
			self.settings.apply_lighting_profile(name)
			self._apply_settings()

	def _on_shader(self, name, index):
		self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
		self._apply_settings()

	def _on_material_prefab(self, name, index):
		self.settings.apply_material_prefab(name)
		self.settings.apply_material = True
		self._apply_settings()

	def _on_material_color(self, color):
		self.settings.material.base_color = [
			color.red, color.green, color.blue, color.alpha
		]
		self.settings.apply_material = True
		self._apply_settings()

	def _on_point_size(self, size):
		self.settings.material.point_size = int(size)
		self.settings.apply_material = True
		self._apply_settings()

	def _on_opacity(self, size):
		print('======================')

		print(self.settings.material)
		print(dir(self.settings.material))

		# for thing in dir(self.settings.material):
		#     print(thing,':', self.settings.material[thing])

		print(self.settings.material.has_alpha)
		print(self.settings.material.transmission)

		print(self.settings.material.shader)
		print(dir(self.settings.material.shader))

		self.settings.material.transmission = 0

		print()
		print('======================')
		# print(self.settings.material.base_color)
		# self.opacity = size / 100.0 * 255
		self.settings.apply_material = True
		self._apply_settings()


	def removeButtonClick(self):
		toRemove = self._model_to_remove_combo.selected_text
		if toRemove == 'No Models Yet':
			return
		self._scene.scene.remove_geometry(toRemove)
		filenameIndex = self.modelNameList.index(toRemove)
		del(self.modelFilenameList[filenameIndex])
		self.modelNameList.remove(toRemove)
		del(self.origionalGeometries[toRemove])
		self._apply_settings()

	def cropButtonClick(self):
		toCrop = self._model_to_remove_combo.selected_text
		if toCrop == 'No Models Yet':
			return
		xmin, ymin, zmin = self.cropMinEdit.vector_value
		xmax, ymax, zmax = self.cropMaxEdit.vector_value

		fullGeometry = self.origionalGeometries[toCrop]

		if hasattr(fullGeometry, 'meshes'): # Mesh
			box = fullGeometry.meshes[0].mesh.get_axis_aligned_bounding_box()

			minBounds = box.get_min_bound()
			minBounds[0] = zmin
			minBounds[1] = xmin
			minBounds[2] = ymin
			maxBounds = box.get_max_bound()
			maxBounds[0] = zmax
			maxBounds[1] = xmax
			maxBounds[2] = ymax
			newBox = o3d.geometry.AxisAlignedBoundingBox(minBounds, maxBounds)

			fullGeometry.meshes[0].mesh = fullGeometry.meshes[0].mesh.crop(newBox)


		else: #Is probably a point cloud?
			box = fullGeometry.get_axis_aligned_bounding_box()

			minBounds = box.get_min_bound()
			minBounds[0] = zmin
			minBounds[1] = xmin
			minBounds[2] = ymin
			maxBounds = box.get_max_bound()
			maxBounds[0] = zmax
			maxBounds[1] = xmax
			maxBounds[2] = ymax
			newBox = o3d.geometry.AxisAlignedBoundingBox(minBounds, maxBounds)

			if newBox.volume() == 0:
				return fullGeometry

			newGeom = fullGeometry.crop(newBox)
			self._scene.scene.remove_geometry(toCrop)
			self._scene.scene.add_geometry(toCrop, newGeom, self.settings.material)
			self._model_to_remove_combo.selected_text = toCrop

			return newGeom

		# newMesh = fullGeometry.meshes[0].mesh.crop(newBox)

		# print('New Mesh Type:', type(newMesh))
		# newGeometry = o3d.visualization.rendering.TriangleMeshModel()
		# print(dir(newGeometry))
		# newGeometry.meshes.append(newMesh)
		# #del(fullGeometry)

		# self._scene.scene.remove_geometry(toCrop)
		# self._scene.scene.add_model(toCrop, newGeometry)

	def uncropButtonClick(self):
		toCrop = self._model_to_remove_combo.selected_text
		if toCrop == 'No Models Yet':
			return
		fullGeometry = self.origionalGeometries[toCrop]

		if hasattr(fullGeometry, 'meshes'):
			pass
		else:
			self._scene.scene.remove_geometry(toCrop)
			self._scene.scene.add_geometry(toCrop, fullGeometry, self.settings.material)

		self.cropMinEdit.vector_value = [0, 0, 0]
		self.cropMaxEdit.vector_value = [0, 0, 0]

	def showHideButtonClick(self):
		toChange = self._model_to_remove_combo.selected_text
		if toChange == 'No Models Yet':
			return
		isVisible = self._scene.scene.geometry_is_visible(toChange)
		self._scene.scene.show_geometry(toChange, not isVisible)

	def getStatsButtonClickedSemantic(self):
		toMeasure = self._model_to_remove_combo.selected_text
		if toMeasure == 'No Models Yet':
			return

		self._on_menu_export_csv()

	def statsButtonSemanticSave(self, csvFilename):
		croppedCloud = self.cropButtonClick()
		xyz_points = np.asarray(croppedCloud.points)
		array = cloudToSemanticArray(xyz_points)
		labels_out = cc3d.connected_components(array, connectivity=26)
		num, count = np.unique(labels_out, return_counts=True)

		countList = count[1:]

		print('H5File Stats')
		print('Min:', min(countList))
		print('Max:', max(countList))
		print('Mean:', np.mean(countList))
		print('Median:', np.median(countList))
		print('Standard Deviation:', np.std(countList))
		print('Sum:', sum(countList))
		print('Total Number:', len(countList))

		countList = list(sorted(countList))

		df2 = pd.DataFrame({"Sample Volume (Voxels)" : countList})
		df2.to_csv(csvFilename)

	def _on_menu_open(self):
		dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose 3D file to load",
							 self.window.theme)
		dlg.add_filter(".ply .stl .fbx .obj .off .gltf .glb .xyz .xyzn .xyzrgb .ply .pcd .pts", "3D Files")
		dlg.add_filter(
			".ply .stl .fbx .obj .off .gltf .glb",
			"Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
			".gltf, .glb)")
		dlg.add_filter(
			".xyz .xyzn .xyzrgb .ply .pcd .pts",
			"Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
			".pcd, .pts)")
		dlg.add_filter(".ply", "Polygon files (.ply)")
		dlg.add_filter(".stl", "Stereolithography files (.stl)")
		dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
		dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
		dlg.add_filter(".off", "Object file format (.off)")
		dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
		dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
		dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
		dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
		dlg.add_filter(".xyzrgb",
					   "ASCII point cloud files with colors (.xyzrgb)")
		dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
		dlg.add_filter(".pts", "3D Points files (.pts)")
		dlg.add_filter("", "All files")

		# A file dialog MUST define on_cancel and on_done functions
		dlg.set_on_cancel(self._on_file_dialog_cancel)
		dlg.set_on_done(self._on_load_dialog_done)
		self.window.show_dialog(dlg)

	def _on_menu_open_tiff(self):
		dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose .tif stack to load",
							 self.window.theme)
		dlg.add_filter(
			".tif .tiff",
			"Tiff Image Stack (.tif, .tiff")
		dlg.add_filter("", "All files")

		# A file dialog MUST define on_cancel and on_done functions
		dlg.set_on_cancel(self._on_file_dialog_cancel)
		dlg.set_on_done(self._on_load_dialog_done_tiff)
		self.window.show_dialog(dlg)

	def _on_file_dialog_cancel(self):
		self.window.close_dialog()

	def _on_load_dialog_done(self, filename):
		self.window.close_dialog()
		self.load(filename)

	def _on_load_dialog_done_tiff(self, filename):
		self.window.close_dialog()
		self.load_tiff(filename)
		self.tiffUpdate()

	def tiffUpdate(self):
		if self.tiffFile == '':
			return

		if self.tiffAxis == 'x':
			self._sliderTiffLocation.set_limits(0, self.tiffShape[0] - 1)
		elif self.tiffAxis == 'y':
			self._sliderTiffLocation.set_limits(0, self.tiffShape[1] - 1)
		elif self.tiffAxis == 'z':
			self._sliderTiffLocation.set_limits(0, self.tiffShape[2] - 1)

		try:
			self._scene.scene.remove_geometry('2D Image Slice')
		except:
			pass
		pcd = getPointCloudImageSliceFromDataset(self.tiffFile, self.tiffAxis, int(self.tiffSliderValue))
		self._scene.scene.add_geometry('2D Image Slice', pcd, self.settings.material)

	def tiffRemove(self):
		self._scene.scene.remove_geometry('2D Image Slice')
		self.tiffFile = ''
		self.tiffShape = [2, 2, 2]

	def tiffXButtonClicked(self):
		self.tiffAxis = 'x'
		self.tiffSliderValue = 0
		self.tiffUpdate()

	def tiffYButtonClicked(self):
		self.tiffAxis = 'y'
		self.tiffSliderValue = 0
		self.tiffUpdate()

	def tiffZButtonClicked(self):
		self.tiffAxis = 'z'
		self.tiffSliderValue = 0
		self.tiffUpdate()

	def tiffSliderChange(self, newValue):
		self.tiffSliderValue = newValue

	def _on_menu_export(self):
		dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
							 self.window.theme)
		dlg.add_filter(".png", "PNG files (.png)")
		dlg.set_on_cancel(self._on_file_dialog_cancel)
		dlg.set_on_done(self._on_export_dialog_done)
		self.window.show_dialog(dlg)

	def _on_menu_export_csv(self):
		dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
							 self.window.theme)
		dlg.add_filter(".csv", "CSV file (.csv)")
		dlg.set_on_cancel(self._on_file_dialog_cancel)
		dlg.set_on_done(self._on_export_dialog_done_csv)
		self.window.show_dialog(dlg)

	def _on_export_dialog_done(self, filename):
		self.window.close_dialog()
		frame = self._scene.frame
		self.export_image(filename, frame.width, frame.height)

	def _on_export_dialog_done_csv(self, filename):
		self.window.close_dialog()
		frame = self._scene.frame
		self.statsButtonSemanticSave(filename)

	def _on_menu_quit(self):
		gui.Application.instance.quit()

	def _on_menu_toggle_settings_panel(self):
		self._settings_panel.visible = not self._settings_panel.visible
		gui.Application.instance.menubar.set_checked(
			AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

	def _on_menu_about(self):
		# Show a simple dialog. Although the Dialog is actually a widget, you can
		# treat it similar to a Window for layout and put all the widgets in a
		# layout which you make the only child of the Dialog.
		em = self.window.theme.font_size
		dlg = gui.Dialog("About")

		# Add the text
		dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
		dlg_layout.add_child(gui.Label("Open3D GUI Example"))

		# Add the Ok button. We need to define a callback function to handle
		# the click.
		ok = gui.Button("OK")
		ok.set_on_clicked(self._on_about_ok)

		# We want the Ok button to be an the right side, so we need to add
		# a stretch item to the layout, otherwise the button will be the size
		# of the entire row. A stretch item takes up as much space as it can,
		# which forces the button to be its minimum size.
		h = gui.Horiz()
		h.add_stretch()
		h.add_child(ok)
		h.add_stretch()
		dlg_layout.add_child(h)

		dlg.add_child(dlg_layout)
		self.window.show_dialog(dlg)

	def _on_about_ok(self):
		self.window.close_dialog()

	def _on_clear(self):
		self._scene.scene.clear_geometry()

		self.modelFilenameList = []
		self.modelNameList = []
		self.origionalGeometries = {}

		self._apply_settings()

	def load(self, path):

		pathHead, pathTail = os.path.split(path)

		geometry = None
		success = False
		geometry_type = o3d.io.read_file_geometry_type(path)

		mesh = None
		if geometry_type & o3d.io.CONTAINS_TRIANGLES:
			mesh = o3d.io.read_triangle_model(path)
		if mesh is None:
			print("[Info]", path, "appears to be a point cloud")
			cloud = None
			try:
				cloud = o3d.io.read_point_cloud(path)
			except Exception:
				pass
			if cloud is not None:
				print("[Info] Successfully read", path)
				if not cloud.has_normals():
					cloud.estimate_normals()
				cloud.normalize_normals()
				geometry = cloud
				success = True
			else:
				print("[WARNING] Failed to read points", path)

		if geometry is not None or mesh is not None:
			try:
				if mesh is not None:
					# Triangle model
					self._scene.scene.add_model(pathTail, mesh)
					success = True
				else:
					# Point cloud
					self._scene.scene.add_geometry(pathTail, geometry,
												   self.settings.material)
					success = True
				bounds = self._scene.scene.bounding_box
				self._scene.setup_camera(60, bounds, bounds.get_center())

			except Exception as e:
				print(e)

		if success:
			self.modelFilenameList.append(path)
			self.modelNameList.append(pathTail)
			if mesh is not None:
				self.origionalGeometries[pathTail] = mesh
			elif geometry is not None:
				self.origionalGeometries[pathTail] = geometry

		self._apply_settings()

	def load_tiff(self, filename):
		self.tiffFile = filename
		self.tiffShape = getShapeOfDataset(filename)
		self.tiffUpdate()

	def export_image(self, path, width, height):

		def on_image(image):
			img = image

			quality = 9  # png
			if path.endswith(".jpg"):
				quality = 100
			o3d.io.write_image(path, img, quality)

		self._scene.scene.scene.render_to_image(on_image)

def runVisualizationWindow():
	# We need to initialize the application, which finds the necessary shaders
	# for rendering and prepares the cross-platform window abstraction.
	gui.Application.instance.initialize()

	w = AppWindow(1024, 768)

	if len(sys.argv) > 1:
		path = sys.argv[1]
		if os.path.exists(path):
			w.load(path)
		else:
			w.window.show_message_box("Error",
									  "Could not open file '" + path + "'")

	# Run the event loop. This will not return until the last window is closed.
	gui.Application.instance.run()

if __name__ == "__main__":
	runVisualizationWindow()