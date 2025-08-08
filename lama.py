#!/usr/bin/env python3
#####################################
###*******************************###
###    Lama bbox Inpainter        ###
###    Author:   Mustapha Ossama  ###
###    Date:     8/8/2025         ###
###*******************************###
#####################################
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
import threading

try:
    from simple_lama_inpainting import SimpleLama
except ImportError:
    SimpleLama = None

class BoundingBoxSelector:
    def __init__(self, canvas, image_width, image_height, scale_factor):
        self.canvas = canvas
        self.image_width = image_width
        self.image_height = image_height
        self.scale_factor = scale_factor
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.bboxes = []
        self.rect_objects = []
        
    def start_selection(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def update_selection(self, event):
        if self.start_x is not None and self.start_y is not None:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            
            self.current_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='red', width=2, tags="bbox"
            )
    
    def end_selection(self, event):
        if self.start_x is not None and self.start_y is not None:
            # Convert canvas coordinates to image coordinates
            x1 = int(min(self.start_x, event.x) / self.scale_factor)
            y1 = int(min(self.start_y, event.y) / self.scale_factor)
            x2 = int(max(self.start_x, event.x) / self.scale_factor)
            y2 = int(max(self.start_y, event.y) / self.scale_factor)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, self.image_width))
            y1 = max(0, min(y1, self.image_height))
            x2 = max(0, min(x2, self.image_width))
            y2 = max(0, min(y2, self.image_height))
            
            if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size check
                self.bboxes.append((x1, y1, x2, y2))
                if self.current_rect:
                    self.rect_objects.append(self.current_rect)
            else:
                if self.current_rect:
                    self.canvas.delete(self.current_rect)
            
            self.start_x = None
            self.start_y = None
            self.current_rect = None
    
    def clear_selections(self):
        for rect in self.rect_objects:
            self.canvas.delete(rect)
        self.bboxes.clear()
        self.rect_objects.clear()

class SimpleLamaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LaMa Bounding Box Inpainting")
        self.root.geometry("1200x800")
        
        self.image = None
        self.photo = None
        self.bbox_selector = None
        self.result_image = None
        self.simple_lama = None
        
        self.setup_ui()
        self.check_dependencies()
        self.initialize_model()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Clear Selections", command=self.clear_selections).pack(side=tk.LEFT, padx=(0, 5))
        
        self.inpaint_btn = ttk.Button(control_frame, text="Inpaint Selected Regions", command=self.start_inpainting)
        self.inpaint_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=(0, 5))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Load an image to get started")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Instructions
        instructions = """I
Note: First run will download the model (~150MB) - please wait!"""
        
        instruction_frame = ttk.Frame(main_frame)
        instruction_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(side=tk.LEFT)
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        if SimpleLama is None:
            messagebox.showerror("Missing Dependency", 
                               "simple-lama-inpainting package is required.\n\n"
                               "Install it with:\n"
                               "pip install simple-lama-inpainting opencv-python Pillow")
            self.root.quit()
            return False
        return True
    
    def initialize_model(self):
        """Initialize the SimpleLama model"""
        def init_model():
            try:
                self.status_var.set("Initializing model (downloading if first run)...")
                self.progress.pack(side=tk.LEFT, padx=(10, 0))
                self.progress.start()
                
                # Force CPU usage for Windows without CUDA
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
                
                self.simple_lama = SimpleLama()
                
                self.progress.stop()
                self.progress.pack_forget()
                self.status_var.set("Model ready! Load an image to get started (CPU mode)")
            except Exception as e:
                self.progress.stop()
                self.progress.pack_forget()
                messagebox.showerror("Model Error", f"Failed to initialize LaMa model:\n{str(e)}")
                self.status_var.set("Model initialization failed")
        
        # Run in separate thread to avoid blocking UI
        threading.Thread(target=init_model, daemon=True).start()
    
    def load_image(self):
        """Load and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        
        if file_path:
            try:
                self.image = Image.open(file_path).convert('RGB')
                self.display_image()
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self):
        """Display image on canvas and setup bbox selector"""
        if self.image is None:
            return
        
        # Calculate display size (max 800x600 while maintaining aspect ratio)
        max_width, max_height = 800, 600
        img_width, img_height = self.image.size
        
        scale_factor = min(max_width / img_width, max_height / img_height, 1.0)
        display_width = int(img_width * scale_factor)
        display_height = int(img_height * scale_factor)
        
        # Resize image for display
        display_image = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Setup bbox selector
        self.bbox_selector = BoundingBoxSelector(
            self.canvas, img_width, img_height, scale_factor
        )
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.bbox_selector.start_selection)
        self.canvas.bind("<B1-Motion>", self.bbox_selector.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.bbox_selector.end_selection)
    
    def clear_selections(self):
        """Clear all bounding box selections"""
        if self.bbox_selector:
            self.bbox_selector.clear_selections()
            self.status_var.set("Selections cleared")
    
    def create_mask_from_bboxes(self, image_shape, bboxes):
        """Create a mask from bounding boxes"""
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for x1, y1, x2, y2 in bboxes:
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def start_inpainting(self):
        """Start inpainting in a separate thread"""
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.bbox_selector or not self.bbox_selector.bboxes:
            messagebox.showwarning("No Selections", "Please select regions to inpaint")
            return
        
        if self.simple_lama is None:
            messagebox.showwarning("Model Not Ready", "Model is still initializing. Please wait.")
            return
        
        # Disable the inpaint button and start progress
        self.inpaint_btn.config(state='disabled')
        self.progress.pack(side=tk.LEFT, padx=(10, 0))
        self.progress.start()
        
        # Run inpainting in separate thread
        threading.Thread(target=self.inpaint_regions, daemon=True).start()
    
    def inpaint_regions(self):
        """Perform inpainting on selected regions using SimpleLama"""
        try:
            self.status_var.set("Processing... Please wait")
            
            # FIXED: Ensure we have a numpy array
            if isinstance(self.image, Image.Image):
                image_np = np.array(self.image, dtype=np.uint8)
            else:
                image_np = self.image.astype(np.uint8)
            
            # Ensure the image is in the correct format (RGB)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # Image is already RGB
                pass
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                # Convert RGBA to RGB
                image_np = image_np[:, :, :3]
            else:
                raise ValueError(f"Unsupported image format: {image_np.shape}")
            
            # Create mask from bounding boxes
            mask = self.create_mask_from_bboxes(image_np.shape, self.bbox_selector.bboxes)
            
            # FIXED: Ensure mask is proper format
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # Debug prints (remove in production)
            print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"Image range: [{image_np.min()}, {image_np.max()}]")
            print(f"Mask range: [{mask.min()}, {mask.max()}]")
            
            result = self.simple_lama(image_np, mask)
            
            if isinstance(result, np.ndarray):
                if result.dtype != np.uint8:
                    if result.max() <= 1.0:
                        # Assuming result is in [0, 1] range
                        result = (result * 255).astype(np.uint8)
                    else:
                        result = result.astype(np.uint8)
                
                if len(result.shape) == 3 and result.shape[2] == 3:
                    self.result_image = Image.fromarray(result, 'RGB')
                elif len(result.shape) == 2:
                    self.result_image = Image.fromarray(result, 'L')
                else:
                    raise ValueError(f"Unexpected result shape: {result.shape}")
                    
            elif isinstance(result, Image.Image):
                # SimpleLama returned a PIL Image directly
                self.result_image = result.convert('RGB')
                print(f"Result is PIL Image: {result.size}, mode: {result.mode}")
                
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")
            
            self.image = self.result_image.copy()  # Update display
            
            self.root.after(0, self.inpainting_completed)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Inpainting error: {error_msg}")  # Debug print
            self.root.after(0, lambda msg=error_msg: self.inpainting_failed(msg))
    
    def inpainting_completed(self):
        """Called when inpainting is completed successfully"""
        self.progress.stop()
        self.progress.pack_forget()
        self.inpaint_btn.config(state='normal')
        
        self.display_image()
        self.clear_selections()
        self.status_var.set(f"Inpainting completed! Processed {len(self.bbox_selector.bboxes)} regions")
    
    def inpainting_failed(self, error_msg):
        """Called when inpainting fails"""
        self.progress.stop()
        self.progress.pack_forget()
        self.inpaint_btn.config(state='normal')
        
        messagebox.showerror("Inpainting Error", f"Failed to perform inpainting:\n{error_msg}")
        self.status_var.set("Inpainting failed")
    
    def save_result(self):
        """Save the inpainted result"""
        if self.result_image is None:
            messagebox.showwarning("No Result", "No inpainted image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Inpainted Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.result_image.save(file_path)
                self.status_var.set(f"Image saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image: {str(e)}")

def main():
    """Main application entry point"""
    # Create and run the application
    root = tk.Tk()
    app = SimpleLamaApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")

if __name__ == "__main__":
    main()