import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
import os
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.spatial import SphericalVoronoi, geometric_slerp
from voronoiGenerator_0 import main_spheroid, generate_random_points_on_sphere, main_2D, preview_voronoi_2D, generate_ellipsoid_points, createSeeds, closest_distance_on_ellipsoid, lloyd_relaxation3D_constrained, evaluate_ellipsoid, is_inside_ellipsoid, is_outside_ellipsoid, project_point_to_ellipsoid_surface
import random
import elasticdeform
import re
import tkinter.font as font
from tkinter import messagebox, filedialog
import yaml
import math

class VoronoiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi Diagram Generator")
        
        # Initialize variables
        self.cellHeight = tk.IntVar(value=75)
        self.cellHeightSlack = tk.IntVar(value=5)
        self.nSeeds = tk.IntVar(value=100)
        self.nSeedsSlack = tk.IntVar(value=10)
        self.ellipsoidAxis1 = tk.IntVar(value=200)
        self.ellipsoidAxis1Slack = tk.IntVar(value=5)
        self.ellipsoidAxis2 = tk.IntVar(value=200)
        self.ellipsoidAxis2Slack = tk.IntVar(value=5)
        self.ellipsoidAxis3 = tk.IntVar(value=200)
        self.ellipsoidAxis3Slack = tk.IntVar(value=5)
        self.matrix_shape = [256, 256, 256]
        self.numImages3D = tk.IntVar(value=1)
        self.numImages3DSlack = tk.IntVar(value=1)
        self.path2save3D = tk.Text(root, height=5, width=100)
        self.path2save3D.insert('1.0', os.getcwd()+'/voronoi3D/') 
        self.path2save2D = tk.Text(root, height=5, width=100)
        self.path2save2D.insert('1.0', os.getcwd()+'/voronoi2D/') 
        self.sideX2D = tk.IntVar(value=256)
        self.sideX2DSlack = tk.IntVar(value=10)
        self.sideY2D = tk.IntVar(value=256)
        self.sideY2DSlack = tk.IntVar(value=10)
        self.numImages2D = tk.IntVar(value=1)
        self.numImages2DSlack = tk.IntVar(value=1)
        self.matrix_shape_2D = [256, 256]
        self.saveName2D = tk.Text(root, height=5, width=100)
        self.saveName2D.insert('1.0', 'voronoi2D') 
        self.saveName3D = tk.Text(root, height=5, width=100)
        self.saveName3D.insert('1.0', 'voronoi3D') 
        self.xsize_var = tk.IntVar(value=512)
        self.ysize_var = tk.IntVar(value=512)
        self.zsize_var = tk.IntVar(value=512)
        self.elasticDeformation = tk.BooleanVar(value=False)
        self.watershedBool = tk.BooleanVar(value=False)
        self.showNuclei = tk.BooleanVar(value=False)
        self.elasticDeformation_value = tk.IntVar(value=1)
        self.lloydIterations_value = tk.IntVar(value=5)
        self.nucleiSize = tk.IntVar(value=5)
        self.lloydIterations_value_3D = tk.IntVar(value=5)
        self.cellDiameter = tk.IntVar(value=20)
        self.cellDiameterSlack = tk.IntVar(value=5)
        self.control_mode = tk.StringVar(value="nSeeds")
        self.slider_widgets_2d = {}
        self.slider_widgets_3d = {}
        
        self.create_widgets()

    def generate_config_string(self):
        result = []
        for attr in dir(self):
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                # Skip attributes that contain the word 'path'
                if 'path' in attr.lower():
                    continue
                
                value = getattr(self, attr)
                
                # Create a short name for each attribute: first letter + capital letters
                short_attr = ''.join([word[0].upper() for word in re.findall(r'[a-zA-Z0-9]+', attr)])
                
                # Handle different types of widgets
                if isinstance(value, (tk.IntVar, tk.BooleanVar)):
                    result.append(f"{short_attr}_{value.get()}")
                elif isinstance(value, tk.Text):
                    result.append(f"{short_attr}_{value.get('1.0', 'end-1c')}")
                elif isinstance(value, list):
                    result.append(f"{short_attr}_{'_'.join(map(str, value))}")
                    
        return '_'.join(result)

    def save_config_to_yaml(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if not file_path:
            return

        config_data = {}
        for attr_name in dir(self):
            if not attr_name.startswith('__') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, tk.IntVar):
                    config_data[attr_name] = attr_value.get()
                elif isinstance(attr_value, tk.BooleanVar):
                    config_data[attr_name] = attr_value.get()
                elif isinstance(attr_value, tk.Text):
                    config_data[attr_name] = attr_value.get('1.0', 'end-1c')
                elif isinstance(attr_value, list):
                    config_data[attr_name] = attr_value
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            messagebox.showinfo("Save Configuration", "Configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Configuration Error", f"Failed to save configuration: {e}")

    def load_config_from_yaml(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)

            for attr_name, value in config_data.items():
                if hasattr(self, attr_name):
                    tk_var = getattr(self, attr_name)
                    if isinstance(tk_var, tk.IntVar):
                        tk_var.set(value)
                    elif isinstance(tk_var, tk.BooleanVar):
                        tk_var.set(value)
                    elif isinstance(tk_var, tk.Text):
                        tk_var.delete('1.0', 'end')
                        tk_var.insert('1.0', value)
                    elif isinstance(tk_var, list):
                        setattr(self, attr_name, value)
            messagebox.showinfo("Load Configuration", "Configuration loaded successfully!")
        except Exception as e:
            messagebox.showerror("Load Configuration Error", f"Failed to load configuration: {e}")

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True)

        # 3D Spheroids Tab
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text='3D Spheroids')

        frame1 = ttk.Frame(tab1)
        frame1.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        radioButtonsFrame = ttk.Frame(frame1)
        radioButtonsFrame.pack(pady=5)
        
        self.nseeds_radio = ttk.Radiobutton(
            radioButtonsFrame, text="Use Number of Seeds", variable=self.control_mode, value="nSeeds", command=self.update_slider_states
        )
        self.nseeds_radio.grid(row=5, column=0, padx=5)

        self.celldiameter_radio = ttk.Radiobutton(
            radioButtonsFrame, text="Use Cell Diameter", variable=self.control_mode, value="cellDiameter", command=self.update_slider_states
        )
        self.celldiameter_radio.grid(row=5, column=1, padx=5)

        self.slider_widgets_3d['nSeeds'] = self.create_slider(frame1, 'Number of Seeds', self.nSeeds, self.nSeedsSlack, 1, 1000)
        self.slider_widgets_3d['cellDiameter'] = self.create_slider(frame1, 'Cell Diameter ', self.cellDiameter, self.cellDiameterSlack, 5, 100)
        
        self.control_mode.set('cellDiameter')
        self.update_slider_states()
        
        
        sliders = [
            ('Cell Height     ', self.cellHeight, self.cellHeightSlack, 10, 1000),
            ('Ellipsoid Axis 1', self.ellipsoidAxis1, self.ellipsoidAxis1Slack, 10, 500),
            ('Ellipsoid Axis 2', self.ellipsoidAxis2, self.ellipsoidAxis2Slack, 10, 500),
            ('Ellipsoid Axis 3', self.ellipsoidAxis3, self.ellipsoidAxis3Slack, 10, 500),
            ('Number of Images', self.numImages3D, None, 1, 500)
        ]
        
        for text, variable, slack_var, min_val, max_val in sliders:
            self.create_slider(frame1, text, variable, slack_var, min_val, max_val)
        
        self.path_entry_3d = ttk.Entry(frame1, textvariable=self.path2save3D, width=80) 
        self.path_entry_3d.insert(0, os.getcwd()+'/voronoi3D/')
        self.path_entry_3d.pack(pady=5)
        
        self.name_entry_3d = ttk.Entry(frame1, textvariable=self.saveName3D, width=80)
        self.name_entry_3d.insert(0, 'savename')
        self.name_entry_3d.pack(pady=5)

        # Frame for buttons
        button_frame = ttk.Frame(frame1)
        button_frame.pack(pady=5)

        self.preview_button = ttk.Button(button_frame, text="Preview Voronoi", command=self.preview_voronoi, width=25)
        self.preview_button.grid(row=0, column=1, padx=5)

        self.generate_button = ttk.Button(button_frame, text="Generate Voronoi", command=self.generate_voronoi, width=25)
        self.generate_button.grid(row=0, column=2, padx=5)

        self.advanced_options_3D_button = ttk.Button(button_frame, text="Advanced Options", command=self.open_advanced_options_3D, width=25)
        self.advanced_options_3D_button.grid(row=0, column=3, padx=5)
        
        # monolayer checkbox
        self.elasticDeformation_checkbox = ttk.Checkbutton(button_frame, text="Elastic Deformation", variable=self.elasticDeformation)
        self.elasticDeformation_checkbox.grid(row=0, column=4, padx=5, sticky='w')

        # monolayer checkbox
        self.watershed_checkbox = ttk.Checkbutton(button_frame, text="Watershed", variable=self.watershedBool)
        self.watershed_checkbox.grid(row=1, column=4, padx=5, sticky='w')

        # Add Save/Load Config buttons for 3D tab
        self.save_config_3d_button = ttk.Button(button_frame, text="Save Config", command=self.save_config_to_yaml, width=25)
        self.save_config_3d_button.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.load_config_3d_button = ttk.Button(button_frame, text="Load Config", command=self.load_config_from_yaml, width=25)
        self.load_config_3d_button.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        
        # Canvas for 3D plot
        self.figure = Figure(figsize=(4, 4), dpi=100)
        # Add black border to the figure
        self.figure.patch.set_edgecolor("#78c2ad")
        self.figure.patch.set_linewidth(5.0)  # Adjust border thickness
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, frame1)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # Progress bar for 3D tab
        self.progress_bar_3d = ttk.Progressbar(frame1, orient="horizontal", mode="determinate")
        self.progress_bar_3d.pack(fill=tk.X, padx=10, pady=10)

        # Status label for 3D tab
        self.status_label_3d = ttk.Label(frame1, text="Idle")
        self.status_label_3d.pack(pady=5)

        # 2D Planes Tab
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text='2D Planes')

        frame2 = ttk.Frame(tab2)
        frame2.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        sliders_frame = ttk.Frame(frame2)
        sliders_frame.pack(pady=5)
        
        radioButtonsFrame = ttk.Frame(frame2)
        radioButtonsFrame.pack(pady=5)
        
        self.nseeds_radio = ttk.Radiobutton(
            radioButtonsFrame, text="Use Number of Seeds", variable=self.control_mode, value="nSeeds", command=self.update_slider_states
        )
        self.nseeds_radio.grid(row=5, column=0, padx=5)

        self.celldiameter_radio = ttk.Radiobutton(
            radioButtonsFrame, text="Use Cell Diameter", variable=self.control_mode, value="cellDiameter", command=self.update_slider_states
        )
        self.celldiameter_radio.grid(row=5, column=1, padx=5)

        self.slider_widgets_2d['nSeeds'] = self.create_slider(frame2, 'Number of Seeds', self.nSeeds, self.nSeedsSlack, 1, 1000)
        self.slider_widgets_2d['cellDiameter'] = self.create_slider(frame2, 'Cell Diameter ', self.cellDiameter, self.cellDiameterSlack, 5, 100)
 
        self.control_mode.set('cellDiameter')
        self.update_slider_states()
        
        slidersTab2 = [
            ('Side X          ', self.sideX2D, self.sideX2DSlack, 10, 1000),
            ('Side Y          ', self.sideY2D, self.sideY2DSlack, 10, 1000),
            ('Number of Images', self.numImages2D, None, 1, 500)]
        
        for text, variable, slack_var, min_val, max_val in slidersTab2:
            self.create_slider(frame2, text, variable, slack_var, min_val, max_val)
        
        preview_generate_frame = ttk.Frame(frame2)
        preview_generate_frame.pack(pady=5)

        self.path_entry_2d = ttk.Entry(frame2, textvariable=self.path2save2D, width=80) 
        self.path_entry_2d.insert(0, os.getcwd()+'/voronoi2D/')
        self.path_entry_2d.pack(pady=5)
        
        self.name_entry_2d = ttk.Entry(frame2, textvariable=self.saveName2D, width=80) 
        self.name_entry_2d.insert(0, 'savename')
        self.name_entry_2d.pack(pady=5)
        
        preview_button_2d = ttk.Button(preview_generate_frame, text="Preview 2D Voronoi", command=self.preview_voronoi_2d, width=25)
        preview_button_2d.grid(row=0, column=0, padx=5)

        generate_button_2d = ttk.Button(preview_generate_frame, text="Generate 2D", command=self.generate_voronoi_2d, width=25)
        generate_button_2d.grid(row=0, column=1, padx=5)
        
        self.advanced_options_2D_button = ttk.Button(preview_generate_frame, text="Advanced Options", command=self.open_advanced_options_2D, width=25)
        self.advanced_options_2D_button.grid(row=0, column=2, padx=5)
        
        # monolayer checkbox
        self.elasticDeformation_checkbox = ttk.Checkbutton(preview_generate_frame, text="Elastic Deformation", variable=self.elasticDeformation)
        self.elasticDeformation_checkbox.grid(row=0, column=3, padx=5, sticky='w')
        
        # monolayer checkbox
        self.elasticDeformation_checkbox = ttk.Checkbutton(preview_generate_frame, text="Show Nuclei", variable=self.showNuclei)
        self.elasticDeformation_checkbox.grid(row=1, column=3, padx=5, pady=5, sticky='w')

        # Add Save/Load Config buttons for 2D tab
        self.save_config_2d_button = ttk.Button(preview_generate_frame, text="Save Config", command=self.save_config_to_yaml, width=25)
        self.save_config_2d_button.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.load_config_2d_button = ttk.Button(preview_generate_frame, text="Load Config", command=self.load_config_from_yaml, width=25)
        self.load_config_2d_button.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        self.canvas_2d = tk.Canvas(
            frame2,
            width=410,
            height=410,
            bg="white",
            highlightthickness=5,
            highlightbackground="#78c2ad",  # Royal purple color
            bd=0
        )
        self.canvas_2d.pack(pady=10)

        # Progress bar for 2D tab
        self.progress_bar_2d = ttk.Progressbar(frame2, orient="horizontal", mode="determinate")
        self.progress_bar_2d.pack(fill=tk.X, padx=10, pady=10)

        # Status label for 2D tab
        self.status_label_2d = ttk.Label(frame2, text="Idle")
        self.status_label_2d.pack(pady=5)

    def create_slider(self, parent, text, variable, slack_variable, min_val, max_val):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        label = ttk.Label(frame, text=text)
        label.pack(side=tk.LEFT, padx=5)
        
        # Fixed width for 4 digits
        num_chars = 4 # Adjusted for better visual alignment with the entry widget
        entry_frame_width = num_chars * 11
        
        # Create a validation command to ensure integer input within range
        vcmd = (self.root.register(self.validate_and_set_slider_value), '%P', variable, min_val, max_val)

        # Create an Entry widget for the main variable
        value_entry_frame = ttk.Frame(frame, relief=tk.SOLID, borderwidth=1, width=entry_frame_width)
        value_entry_frame.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.Y)
        value_entry = ttk.Entry(value_entry_frame, textvariable=variable, width=num_chars, validate="focusout", validatecommand=vcmd)
        value_entry.pack(expand=True, fill=tk.BOTH)
        value_entry_frame.config(width=entry_frame_width) # Set the frame width
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=variable, command=lambda v: variable.set(int(float(v))))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        slider.config(length=400) # Set a fixed length for the scale bar
        
        slack_label = None
        slack_value_entry = None

        if slack_variable is not None:
            slack_label = ttk.Label(frame, text="±")
            slack_label.pack(side=tk.LEFT, padx=5)

            # Create a validation command for the slack variable
            # Assuming slack values generally range from 0 to 20, adjust if needed
            slack_vcmd = (self.root.register(self.validate_and_set_slider_value), '%P', slack_variable, 0, 20)
            
            # Create an Entry widget for the slack variable
            slack_value_entry_frame = ttk.Frame(frame, relief=tk.SOLID, borderwidth=1, width=entry_frame_width)
            slack_value_entry_frame.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.Y)
            slack_value_entry = ttk.Entry(slack_value_entry_frame, textvariable=slack_variable, width=num_chars, validate="focusout", validatecommand=slack_vcmd)
            slack_value_entry.pack(expand=True, fill=tk.BOTH)
            slack_value_entry_frame.config(width=entry_frame_width)

        # Return a dictionary of the created widgets for external control
        return {
            'label': label,
            'entry': value_entry,
            'slider': slider,
            'slack_label': slack_label,
            'slack_entry': slack_value_entry
        }

    # Add this new method to your VoronoiApp class to handle input validation for Entry widgets
    def validate_and_set_slider_value(self, P, tk_var, min_val, max_val):
        """
        Validates the input P for an Entry widget.
        P: The value of the Entry widget if the edit is allowed.
        tk_var: The associated Tkinter variable (tk.IntVar).
        min_val: The minimum allowed value for the slider.
        max_val: The maximum allowed value for the slider.
        """
        if P.strip() == "": # Allow empty input temporarily during typing
            return True
        try:
            value = int(P) # Try to convert input to an integer
            if min_val <= value <= max_val: # Check if value is within bounds
                tk_var.set(value) # Set the Tkinter variable
                return True # Validation successful
            else:
                messagebox.showwarning("Input Error", f"Value must be between {min_val} and {max_val}.")
                # Reset the Entry to the current valid value if input is out of range
                # Use self.root.after(0, ...) to ensure the update happens after validation
                self.root.after(0, lambda: tk_var.set(tk_var.get()))
                return False # Validation failed
        except ValueError: # If input is not a valid integer
            messagebox.showwarning("Input Error", "Please enter an integer number.")
            # Reset the Entry to the current valid value if input is not an integer
            self.root.after(0, lambda: tk_var.set(tk_var.get()))
            return False # Validation failed


    def open_advanced_options_3D(self):
        top = tk.Toplevel(self.root)
        top.title("Advanced Options")

        sliders_popup = [
            ("zsize", self.zsize_var, None, 10, 1000),
            ("xsize", self.xsize_var, None, 10, 1000),
            ("ysize", self.ysize_var, None, 10, 1000),
            ('Number of Seeds ', self.nSeeds, self.nSeedsSlack, 10, 5000),
            ("elastic deformation", self.elasticDeformation_value, None, 1, 50),
            ("lloyd iterations", self.lloydIterations_value_3D, None, 0, 10)
        ]
        
        frame_popup = ttk.Frame(top)
        frame_popup.pack(fill=tk.X, pady=5)
        for text, variable, slack_var, min_val, max_val in sliders_popup:
            self.create_slider(frame_popup, text, variable, slack_var, min_val, max_val)

        button_frame = ttk.Frame(top)
        button_frame.pack(pady=10)

        ok_button = ttk.Button(button_frame, text="Ok", command=lambda: self.save_advanced_options_3D(top))
        ok_button.pack(side=tk.LEFT, padx=5)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=top.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

    def save_advanced_options_3D(self, top):
        self.matrix_shape = [
            self.xsize_var.get(),
            self.ysize_var.get(),
            self.zsize_var.get()
            ]
        top.destroy()
        
    def open_advanced_options_2D(self):
        top = tk.Toplevel(self.root)
        top.title("Advanced Options")

        sliders_popup = [
            ("xsize", self.xsize_var, None, 10, 1000),
            ("ysize", self.ysize_var, None, 10, 1000),
            ('Number of Seeds ', self.nSeeds, self.nSeedsSlack, 10, 5000),
            ("elastic deformation", self.elasticDeformation_value, None, 1, 50),
            ("lloyd iterations", self.lloydIterations_value, None, 0, 10),
            ("nuclei size", self.nucleiSize, None, 1, 10)
        ]
        
        frame_popup = ttk.Frame(top)
        frame_popup.pack(fill=tk.X, pady=5)
        for text, variable, slack_var, min_val, max_val in sliders_popup:
            self.create_slider(frame_popup, text, variable, slack_var, min_val, max_val)

        button_frame = ttk.Frame(top)
        button_frame.pack(pady=10)

        ok_button = ttk.Button(button_frame, text="Ok", command=lambda: self.save_advanced_options_2D(top))
        ok_button.pack(side=tk.LEFT, padx=5)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=top.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

    @staticmethod
    def estimate_hexagons_on_spheroid(axis1, axis2, axis3, diameter, packing_efficiency=0.9):
        """
        Estimate how many hexagons of flat-to-flat diameter D fit on a spheroid surface.
        """
        # Step 1: Estimate surface area of ellipsoid (Knud Thomsen's formula)
        p = 1.6075
        a, b, c = axis1, axis2, axis3
        term = ((a**p * b**p) + (a**p * c**p) + (b**p * c**p)) / 3
        surface_area = 4 * math.pi * (term ** (1/p))
        
        # Step 2: Compute hexagon area from flat-to-flat diameter
        s = diameter / 2  # side length
        hex_area = (3 * math.sqrt(3) / 2) * s**2
    
        # Step 3: Estimate number of hexagons
        estimated_count = (surface_area * packing_efficiency) / hex_area
        return int(estimated_count)
        
    def update_slider_states(self):
        """
        Updates the state of the nSeeds and cellDiameter sliders for both 2D and 3D modes,
        based on the selected control mode.
        """
        mode = self.control_mode.get()
    
        # List of widget dictionaries to process (2D and 3D)
        for widget_group in [self.slider_widgets_2d, self.slider_widgets_3d]:
            nSeeds_widgets = widget_group.get('nSeeds')
            cellDiameter_widgets = widget_group.get('cellDiameter')
    
            if not nSeeds_widgets or not cellDiameter_widgets:
                continue  # skip if either is missing
    
            # Define active/inactive widget sets based on mode
            if mode == "nSeeds":
                active_widgets = nSeeds_widgets
                inactive_widgets = cellDiameter_widgets
            elif mode == "cellDiameter":
                active_widgets = cellDiameter_widgets
                inactive_widgets = nSeeds_widgets
            else:
                continue  # skip unknown modes
    
            # Set active widgets to enabled / black
            for key, widget in active_widgets.items():
                if 'label' in key:
                    widget.config(foreground='black')
                else:
                    widget.config(state='normal')
    
            # Set inactive widgets to disabled / gray
            for key, widget in inactive_widgets.items():
                if 'label' in key:
                    widget.config(foreground='gray')
                else:
                    widget.config(state='disabled')

                
    def save_advanced_options_2D(self, top):
        self.matrix_shape_2D = [
            self.xsize_var.get(),
            self.ysize_var.get(),
            ]
        top.destroy()
    
    def preview_voronoi(self):
        self.ax.clear()
        self.status_label_3d.config(text="Busy")
        self.progress_bar_3d.start()
        self.root.update_idletasks()
    
        try:
            cell_height_val = self.cellHeight.get()
            n_seeds_val = self.nSeeds.get()
            ax1_val = self.ellipsoidAxis1.get()
            ax2_val = self.ellipsoidAxis2.get()
            ax3_val = self.ellipsoidAxis3.get()
            lloyd_iters_val = self.lloydIterations_value_3D.get()
            
            if self.control_mode.get()=='cellDiameter':
                n_seeds_val = self.estimate_hexagons_on_spheroid(sideAxis1, sideAxis2, sideAxis3, cellDiameter)
    
            # Retrieve image dimensions
            xsize_val = self.xsize_var.get()
            ysize_val = self.ysize_var.get()
            zsize_val = self.zsize_var.get()
    
            min_axis_length = min(ax1_val, ax2_val, ax3_val)
            if cell_height_val >= min_axis_length:
                messagebox.showwarning(
                    "Input Error",
                    f"Cell height ({cell_height_val}) cannot be greater than or equal to "
                    f"the smallest ellipsoid axis half length ({min_axis_length})."
                )
                self.status_label_3d.config(text="Error")
                self.progress_bar_3d.stop()
                return
    
            # New Warning Check: Spheroid size vs Image size
            if (ax1_val > xsize_val or
                ax2_val > ysize_val or
                ax3_val > zsize_val):
                messagebox.showwarning(
                    "Warning!",
                    f"Ellipsoid dimensions ({ax1_val}, {ax2_val}, {ax3_val}) exceed "
                    f"image dimensions ({xsize_val}, {ysize_val}, {zsize_val}). "
                    "This may cause the spheroid to be cut off. You can go ahead if you don't mind!"
                )
                # The preview will continue after this warning.
    
            outer_radii = np.array([ax1_val, ax2_val, ax3_val])
            inner_radii = np.array([ax1_val - cell_height_val, ax2_val - cell_height_val, ax3_val - cell_height_val])
            seed_placement_radii = outer_radii - cell_height_val / 2
    
            max_dim_for_lloyd = int(max(outer_radii) * 2 + 10)
            dim_x, dim_y, dim_z = max_dim_for_lloyd, max_dim_for_lloyd, max_dim_for_lloyd
            center = np.array([round(dim_x / 2), round(dim_y / 2), round(dim_z / 2)])
    
            self.status_label_3d.config(text="Generating initial seeds...")
            self.progress_bar_3d['value'] = 10
            self.root.update_idletasks()
    
            coords_seed_placement = generate_ellipsoid_points(center, seed_placement_radii, resolution=500)
    
            vol_ellipsoid = (4/3) * np.pi * np.prod(seed_placement_radii)
            vol_per_seed = vol_ellipsoid / n_seeds_val if n_seeds_val > 0 else 1.0
            min_sep_initial = max(1, int(vol_per_seed**(1/3.0)))
    
            central_idx = np.array([], dtype=int)
            attempts = 0
            max_attempts = 50
            current_min_sep = float(min_sep_initial)
    
            while len(central_idx) < n_seeds_val and attempts < max_attempts:
                current_min_sep = max(0.1, current_min_sep)
                try:
                    temp_central_idx = createSeeds(n_seeds_val, coords_seed_placement, current_min_sep)
                    if temp_central_idx is not None and len(temp_central_idx) > len(central_idx):
                        central_idx = temp_central_idx
                        if len(central_idx) >= n_seeds_val:
                            break
                except ValueError:
                    pass
    
                if len(central_idx) < n_seeds_val / 2:
                    current_min_sep *= 0.8
                elif len(central_idx) < n_seeds_val * 0.9:
                    current_min_sep *= 0.9
                else:
                    current_min_sep -= 0.1
    
                current_min_sep = max(0.01, current_min_sep)
                attempts += 1
                if attempts % 5 == 0 or attempts == max_attempts - 1:
                    print(f"Retrying seed creation with min_sep = {current_min_sep:.2f} (Attempt {attempts}/{max_attempts}). Current seeds: {len(central_idx)}")
    
            if len(central_idx) == 0:
                messagebox.showwarning("Seed Generation Error", "Could not generate any seeds. Adjust parameters.")
                self.status_label_3d.config(text="Error")
                self.progress_bar_3d.stop()
                return
    
            if len(central_idx) < n_seeds_val:
                messagebox.showwarning("Seed Generation Warning", f"Generated only {len(central_idx)} seeds.")
    
            central_points = coords_seed_placement[central_idx]
            self.status_label_3d.config(text=f"Generated {len(central_points)} seeds. Applying Lloyd relaxation...")
            self.progress_bar_3d['value'] = 30
            self.root.update_idletasks()
    
            relaxed_points, _ = lloyd_relaxation3D_constrained(
                central_points, lloyd_iters_val, dim_x, dim_y, dim_z,
                ellipsoid_center=center,
                ellipsoid_radii_inner=inner_radii,
                ellipsoid_radii_outer=outer_radii
            )
    
            self.status_label_3d.config(text="Filtering and adjusting seeds for preview...")
            self.progress_bar_3d['value'] = 60
            self.root.update_idletasks()
    
            inside_outer_mask = is_inside_ellipsoid(relaxed_points, center, outer_radii)
            outside_inner_mask = is_outside_ellipsoid(relaxed_points, center, inner_radii)
            filtered_points = relaxed_points[inside_outer_mask & outside_inner_mask]
    
            adjusted_points = np.copy(filtered_points)
            epsilon = 1e-6
    
            outer_eval = np.array([evaluate_ellipsoid(p, center, outer_radii) for p in adjusted_points])
            close_to_outer = (outer_eval > (1.0 - epsilon)) & (outer_eval <= 1.0)
            for i in np.where(close_to_outer)[0]:
                projected_p = project_point_to_ellipsoid_surface(adjusted_points[i], center, outer_radii)
                adjusted_points[i] = center + (projected_p - center) * (1.0 - epsilon)
    
            inner_eval = np.array([evaluate_ellipsoid(p, center, inner_radii) for p in adjusted_points])
            close_to_inner = (inner_eval >= 1.0) & (inner_eval < (1.0 + epsilon))
            for i in np.where(close_to_inner)[0]:
                projected_p = project_point_to_ellipsoid_surface(adjusted_points[i], center, inner_radii)
                adjusted_points[i] = center + (projected_p - center) * (1.0 + epsilon)
    
            final_points = adjusted_points
    
            if len(final_points) < 4:
                messagebox.showwarning("Preview Error", "Not enough seeds remained to form a Voronoi diagram.")
                self.status_label_3d.config(text="Done!")
                self.progress_bar_3d.stop()
                return
    
            # FIX: center points around origin before normalization
            centered_points = final_points - center
            norms = np.linalg.norm(centered_points, axis=1)
            norms[norms == 0] = 1e-8
            unit_sphere_points = centered_points / norms[:, np.newaxis]
    
            self.status_label_3d.config(text="Drawing Voronoi diagram...")
            self.progress_bar_3d['value'] = 80
            self.root.update_idletasks()
    
            sv = SphericalVoronoi(unit_sphere_points, radius=1, center=[0, 0, 0])
            sv.sort_vertices_of_regions()
    
            elongation_matrix = np.diag(outer_radii)
    
            # Define resolution for plotting spheres
            sphere_plot_resolution_u = 100
            sphere_plot_resolution_v = 100
    
            u = np.linspace(0, 2 * np.pi, sphere_plot_resolution_u)
            v = np.linspace(0, np.pi, sphere_plot_resolution_v)
            x_unit = np.outer(np.cos(u), np.sin(v))
            y_unit = np.outer(np.sin(u), np.sin(v))
            z_unit = np.outer(np.ones(np.size(u)), np.cos(v))
    
            # Plot the INNER Sphere as a smooth surface (no grid)
            self.ax.plot_surface(inner_radii[0] * x_unit, inner_radii[1] * y_unit, inner_radii[2] * z_unit,
                                 color='lightgray',
                                 alpha=0.3,
                                 edgecolor='none',
                                 rstride=1,
                                 cstride=1)
    
            # Plot the OUTER Ellipsoid (Boundary for Voronoi) as a subtle wireframe
            self.ax.plot_wireframe(outer_radii[0] * x_unit, outer_radii[1] * y_unit, outer_radii[2] * z_unit,
                                 color='gray',
                                 alpha=0.2,
                                 linewidth=0.5)
    
            # Plot the cubic shape in semitransparent red
            x_coords = np.array([-xsize_val, xsize_val])
            y_coords = np.array([-ysize_val, ysize_val])
            z_coords = np.array([-zsize_val, zsize_val])
    
            # Faces of the cube
            # Front face (z_max)
            X, Y = np.meshgrid(x_coords, y_coords)
            Z = np.full_like(X, z_coords[1])
            self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.1, edgecolor='r', linewidth=0.5)
    
            # Back face (z_min)
            X, Y = np.meshgrid(x_coords, y_coords)
            Z = np.full_like(X, z_coords[0])
            self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.1, edgecolor='r', linewidth=0.5)
    
            # Left face (x_min)
            Y, Z = np.meshgrid(y_coords, z_coords)
            X = np.full_like(Y, x_coords[0])
            self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.1, edgecolor='r', linewidth=0.5)
    
            # Right face (x_max)
            Y, Z = np.meshgrid(y_coords, z_coords)
            X = np.full_like(Y, x_coords[1])
            self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.1, edgecolor='r', linewidth=0.5)
    
            # Top face (y_max)
            X, Z = np.meshgrid(x_coords, z_coords)
            Y = np.full_like(X, y_coords[1])
            self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.1, edgecolor='r', linewidth=0.5)
    
            # Bottom face (y_min)
            X, Z = np.meshgrid(x_coords, z_coords)
            Y = np.full_like(X, y_coords[0])
            self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.1, edgecolor='r', linewidth=0.5)
    
            t_vals = np.linspace(0, 1, 2000)
            for region in sv.regions:
                if not region:
                    continue
                n = len(region)
                for i in range(n):
                    start_vertex = sv.vertices[region][i]
                    end_vertex = sv.vertices[region][(i + 1) % n]
                    result = geometric_slerp(start_vertex, end_vertex, t_vals)
                    transformed_result = result @ elongation_matrix.T
                    self.ax.plot(transformed_result[:, 0], transformed_result[:, 1], transformed_result[:, 2], c='k', linewidth=0.5)
    
            max_axis_plot_limit = max(max(outer_radii), xsize_val, ysize_val, zsize_val) * 1.1 # Extend limits slightly
            self.ax.set_xlim(-max_axis_plot_limit, max_axis_plot_limit)
            self.ax.set_ylim(-max_axis_plot_limit, max_axis_plot_limit)
            self.ax.set_zlim(-max_axis_plot_limit, max_axis_plot_limit)
            self.ax.set_box_aspect([1,1,1]) # Set equal aspect ratio for 3D plots
            self.ax.set_title('3D Voronoi Preview')
            self.ax.set_xlabel('X-axis')
            self.ax.set_ylabel('Y-axis')
            self.ax.set_zlabel('Z-axis')
    
            self.canvas.draw()
            self.status_label_3d.config(text=f"Done! ({len(final_points)} seeds)")
    
        except Exception as e:
            messagebox.showerror("Preview Error", f"An error occurred during 3D preview: {e}")
            self.status_label_3d.config(text="Error")
            print(f"Error in preview_voronoi: {e}")
    
        self.progress_bar_3d.stop()

    def generate_voronoi(self):
        self.status_label_3d.config(text="Busy")
        self.root.update_idletasks()
        self.progress_bar_3d.start()
        
        step = 0
        steps = self.numImages3D.get()
        
        for image in range(self.numImages3D.get()):
            
            # slack (+-)
            
            slackCellHeight = random.uniform(-self.cellHeightSlack.get(), self.cellHeightSlack.get())
            slackNumOfSeeds = random.uniform(-self.nSeedsSlack.get(), self.nSeedsSlack.get())
            slackAxis1 = random.uniform(-self.ellipsoidAxis1Slack.get(), self.ellipsoidAxis1Slack.get())
            slackAxis2 = random.uniform(-self.ellipsoidAxis2Slack.get(), self.ellipsoidAxis2Slack.get())
            slackAxis3 = random.uniform(-self.ellipsoidAxis3Slack.get(), self.ellipsoidAxis3Slack.get())
            slackCellDiameter = random.uniform(-self.cellDiameterSlack.get(), self.cellDiameterSlack.get())

            # Modify the value by this random quantity
            cellHeight = np.round(self.cellHeight.get() + slackNumOfSeeds)
            nSeeds = np.round(self.nSeeds.get() + slackNumOfSeeds)
            sideAxis1 = np.round(self.ellipsoidAxis1.get() + slackAxis1)
            sideAxis2 = np.round(self.ellipsoidAxis2.get() + slackAxis2)
            sideAxis3 = np.round(self.ellipsoidAxis3.get() + slackAxis3)
            cellDiameter = np.round(self.cellDiameter.get() + slackCellDiameter)
            
            if self.control_mode.get()=='cellDiameter':
                nSeeds = self.estimate_hexagons_on_spheroid(sideAxis1, sideAxis2, sideAxis3, cellDiameter)

            
            
            # Strip newline characters from the strings retrieved from the text widgets
            path2save = self.path_entry_3d.get()
            saveName = self.generate_config_string()
            # Ensure the directory exists
            if not os.path.exists(path2save):
                os.makedirs(path2save)
            
            # Construct the file path
            file_path = os.path.join(path2save, f"{saveName}_{image}.tiff")
            
            # Update the status label and save the image
            message = f"saving to {file_path}"
            self.status_label_3d.config(text=message)

            try:
                # Call the main_3D function with the constructed file path
                main_spheroid(
                    cellHeight, 
                    nSeeds, 
                    sideAxis1, 
                    sideAxis2, 
                    sideAxis3, 
                    self.matrix_shape,
                    file_path,
                    self.elasticDeformation.get(),
                    self.elasticDeformation_value.get(),
                    self.lloydIterations_value_3D.get(),
                    self.watershedBool.get())
            except ValueError as e:
                messagebox.showerror("Generation Error", f"Error generating image:\n{e}\n\nGeneration stopped.")
                self.status_label_3d.config(text=f"Generation stopped. Try again")
                self.progress_bar_3d.stop()
                self.progress_bar_3d['value'] = 0
                return
            except Exception as e:
                messagebox.showerror("Generation Error", f"An error occurred:\n{e}\n\nGeneration stopped.")
                self.status_label_3d.config(text=f"Error: {e}")
                self.progress_bar_3d.stop()
                self.progress_bar_3d['value'] = 0
                return
            
            # Update the progress bar
            self.progress_bar_3d['value'] = (image + 1) / self.numImages3D.get() * 100
            self.progress_bar_3d.update_idletasks()
        self.status_label_3d.config(text="Done!")
        self.progress_bar_3d.stop()
        self.save_config_to_yaml()


    def preview_voronoi_2d(self):
        self.ax.clear()
        step = 0
        steps = 100
        self.progress_bar_2d.start()

        self.status_label_2d.config(text="Busy")
        self.root.update_idletasks()
        self.progress_bar_2d['value'] = step+10 / steps * 100
        self.progress_bar_2d.update_idletasks()
        # Generate the Voronoi diagram as a 2D matrix
        
        nSeeds = self.nSeeds.get()
        if self.control_mode.get() == 'cellDiameter':
            nSeeds = self.sideX2D.get()/self.cellDiameter.get()*self.sideY2D.get()/self.cellDiameter.get()

        voronoi_image = preview_voronoi_2D(nSeeds, self.sideX2D.get(), self.sideY2D.get(), self.matrix_shape_2D, progress_bar_2d=self.progress_bar_2d, elasticDeformation=self.elasticDeformation.get(), elasticDeformation_value=self.elasticDeformation_value.get(), showNuclei=self.showNuclei.get(), nucleiSize=self.nucleiSize.get(), lloydIters=self.lloydIterations_value.get())
        self.progress_bar_2d['value'] = step+10 / steps * 100
        self.progress_bar_2d.update_idletasks()

        voronoi_image = voronoi_image.astype(np.uint8)
        image = Image.fromarray(voronoi_image)
        image = image.resize((400, 400), Image.LANCZOS)
        self.progress_bar_2d['value'] = step+10 / steps * 100
        self.progress_bar_2d.update_idletasks()
        

        # Convert the image to a format that can be displayed in Tkinter
        self.tk_image = ImageTk.PhotoImage(image)
        
        
        # Display the image on the canvas
        self.canvas_2d.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.progress_bar_2d['value'] = step+10 / steps * 100
        self.progress_bar_2d.update_idletasks()
        
        self.progress_bar_2d.stop()

        self.status_label_2d.config(text="Done!")

    def generate_voronoi_2d(self):
        self.status_label_2d.config(text="Busy")
        self.root.update_idletasks()
        self.progress_bar_2d.start()
        
        step = 0
        steps = self.numImages2D.get()
        
        for image in range(self.numImages2D.get()):
            
            # slack (+-)
            
            slackSideX = random.uniform(-self.sideX2DSlack.get(), self.sideX2DSlack.get())
            slackSideY = random.uniform(-self.sideY2DSlack.get(), self.sideY2DSlack.get())
            slackNumOfSeeds = random.uniform(-self.nSeedsSlack.get(), self.nSeedsSlack.get())

            # Modify the value by this random quantity
            sideX2D = np.round(self.sideX2D.get() + slackSideX)
            sideY2D = np.round(self.sideY2D.get() + slackSideY)
            nSeeds = np.round(self.nSeeds.get() + slackNumOfSeeds)
            
            # Strip newline characters from the strings retrieved from the text widgets
            path2save = self.path_entry_2d.get()
            saveName = self.name_entry_2d.get()
            
            # Ensure the directory exists
            if not os.path.exists(path2save):
                os.makedirs(path2save)
            
            # Construct the file path
            file_path = os.path.join(path2save, f"{saveName}_{image}.tiff")
            
            # Update the status label and save the image
            message = f"saving to {file_path}"
            self.status_label_2d.config(text=message)
            
            # Call the main_2D function with the constructed file path
            # main_2D(nSeeds, sideX2D, sideY2D, self.matrix_shape_2D, file_path)
            
            if self.control_mode.get() == 'cellDiameter':
                nSeeds = sideX2D/self.cellDiameter.get()*sideY2D/self.cellDiameter.get()

            main_2D(nSeeds, sideX2D, sideY2D, self.matrix_shape_2D, file_path, elasticDeformation=self.elasticDeformation.get(), elasticDeformation_value=self.elasticDeformation_value.get(), showNuclei=self.showNuclei.get(), lloydIters=self.lloydIterations_value.get())

            # Update the progress bar
            self.progress_bar_2d['value'] = (image + 1) / self.numImages2D.get() * 100
            self.progress_bar_2d.update_idletasks()
        self.status_label_2d.config(text="Done!")
        self.progress_bar_2d.stop()
        self.save_config_to_yaml()
    

if __name__ == "__main__":
    root = ttk.Window()
    style = ttk.Style()
    
    try:
        style.load_user_themes("scutoidTheme.json")
        style.theme_use("scutoidTheme")
        print("Attempted to load and use custom theme 'scutoidTheme.json'")
    except Exception as e:
        print(f"Error loading custom theme 'scutoidTheme.json': {e}")
        messagebox.showerror("Theme Error", f"Could not load custom theme 'scutoidTheme.json'. Error: {e}")
        style.theme_use("default")

    print(f"Current active theme: {style.theme_use()}")
    print(f"Available themes: {style.theme_names()}")

    app = VoronoiApp(root)
    root.mainloop()
