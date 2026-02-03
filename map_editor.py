# Editor to make customized TravelField maps.

import tkinter as tk
from tkinter import ttk, filedialog
import torch

# Terrain constants
GRASS = 1
SWAMP = 2
ROCK = 3
ROAD = 4
GOAL = 5

char_dict = {
    ".": GRASS,
    "□": ROCK,
    "+": ROAD,
    "-": SWAMP,
    "O": GOAL,
}

inv_char_dict = {v: k for k, v in char_dict.items()}


class MapEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Map Editor")

        self.grid_size = 32
        self.cell_size = 15

        self.terrain_types = {
            GRASS: {"name": "Grass", "color": (0, 255, 0)},
            SWAMP: {"name": "Swamp", "color": (139, 69, 19)},
            ROCK: {"name": "Rock", "color": (100, 100, 100)},
            ROAD: {"name": "Road", "color": (255, 255, 155)},
            GOAL: {"name": "Goal", "color": (255, 0, 0)},
        }

        self.grid = torch.zeros((8, self.grid_size, self.grid_size))
        self.undo_stack = []

        self.is_painting = False
        self.is_erasing = False

        self.setup_gui()
        self.initialize_borders()
        self.update_window_size()
        self.draw_grid()

    # ------------------- GUI -------------------
    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Row 1: Terrain, Map size, Apply, Pan buttons
        control_frame1 = ttk.Frame(self.main_frame)
        control_frame1.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(control_frame1, text="Terrain:").pack(side=tk.LEFT)
        self.terrain_var = tk.StringVar()
        self.terrain_dropdown = ttk.Combobox(
            control_frame1,
            textvariable=self.terrain_var,
            values=[f"{k}: {v['name']}" for k, v in self.terrain_types.items()],
            state="readonly",
            width=14
        )
        self.terrain_dropdown.set("1: Grass")
        self.terrain_dropdown.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(control_frame1, text="Map Size:").pack(side=tk.LEFT)
        self.size_var = tk.IntVar(value=self.grid_size)
        ttk.Spinbox(control_frame1, from_=4, to=256, textvariable=self.size_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame1, text="Apply", command=self.apply_size).pack(side=tk.LEFT, padx=(5, 20))

        # Pan buttons
        pan_frame = ttk.Frame(control_frame1)
        pan_frame.pack(side=tk.LEFT)
        ttk.Button(pan_frame, text="▲", width=3, command=lambda: self.pan_canvas(0, -5)).pack()
        mid = ttk.Frame(pan_frame)
        mid.pack()
        ttk.Button(mid, text="◀", width=3, command=lambda: self.pan_canvas(-5, 0)).pack(side=tk.LEFT)
        ttk.Button(mid, text="▶", width=3, command=lambda: self.pan_canvas(5, 0)).pack(side=tk.LEFT)
        ttk.Button(pan_frame, text="▼", width=3, command=lambda: self.pan_canvas(0, 5)).pack()

        # Row 2: Undo, Restart, Save, Load
        control_frame2 = ttk.Frame(self.main_frame)
        control_frame2.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(control_frame2, text="Undo", command=self.undo_last).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame2, text="Restart", command=self.restart_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame2, text="Save", command=self.save_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame2, text="Load", command=self.load_map).pack(side=tk.LEFT, padx=5)

        # Canvas frame
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbars (created once, updated dynamically)
        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        # Mouse bindings
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_mouse_down)
        self.canvas.bind("<B3-Motion>", self.on_right_mouse_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_mouse_up)
        self.canvas.bind("<Button-2>", self.on_middle_mouse_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_mouse_drag)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X)

    # ------------------- Map Logic -------------------
    def initialize_borders(self):
        self.grid.zero_()
        self.grid[GRASS, :, :] = 1.0

    def apply_size(self):
        n = int(self.size_var.get())
        self.grid_size = n
        self.grid = torch.zeros((8, n, n))
        self.initialize_borders()
        self.update_window_size()
        self.draw_grid()

    def update_window_size(self):
        """Resize window and canvas to fit map + controls, add scrollbars only if needed."""

        # Canvas size
        max_width = self.root.winfo_screenwidth() - 100
        max_height = self.root.winfo_screenheight() - 150

        # Minimum canvas size so buttons always visible
        min_canvas_width = 300
        min_canvas_height = 200

        canvas_width = max(min(self.grid_size * self.cell_size, max_width), min_canvas_width)
        canvas_height = max(min(self.grid_size * self.cell_size, max_height), min_canvas_height)

        self.canvas.config(width=canvas_width, height=canvas_height)

        # Scrollbars if needed
        if self.grid_size * self.cell_size > canvas_width:
            self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self.h_scroll.pack_forget()
        if self.grid_size * self.cell_size > canvas_height:
            self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.v_scroll.pack_forget()

        # Measure actual control rows + status bar height
        self.root.update_idletasks()  # ensures sizes are computed
        control_height = (
            self.main_frame.winfo_children()[0].winfo_reqheight() +  # first row
            self.main_frame.winfo_children()[1].winfo_reqheight() +  # second row
            self.main_frame.winfo_children()[-1].winfo_reqheight() + 10  # status bar + small padding
        )

        # Set window geometry
        window_width = canvas_width + 40
        window_height = canvas_height + control_height
        self.root.geometry(f"{window_width}x{window_height}")

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                t = torch.argmax(self.grid[:, i, j]).item()
                rgb = self.terrain_types[t]["color"]
                color = "#%02x%02x%02x" % rgb
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # ------------------- Undo / Paint -------------------
    def push_undo(self):
        self.undo_stack.append(self.grid.clone())

    def paint_cell(self, event, terrain_type):
        x = int(self.canvas.canvasx(event.x) // self.cell_size)
        y = int(self.canvas.canvasy(event.y) // self.cell_size)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[:, y, x] = 0
            self.grid[terrain_type, y, x] = 1
            self.draw_grid()

    # ------------------- Mouse events -------------------
    def on_mouse_down(self, event):
        self.push_undo()
        self.is_painting = True
        terrain_type = int(self.terrain_var.get().split(":")[0])
        self.paint_cell(event, terrain_type)

    def on_mouse_drag(self, event):
        if self.is_painting:
            terrain_type = int(self.terrain_var.get().split(":")[0])
            self.paint_cell(event, terrain_type)

    def on_mouse_up(self, event):
        self.is_painting = False

    def on_right_mouse_down(self, event):
        self.push_undo()
        self.is_erasing = True
        self.paint_cell(event, GRASS)

    def on_right_mouse_drag(self, event):
        if self.is_erasing:
            self.paint_cell(event, GRASS)

    def on_right_mouse_up(self, event):
        self.is_erasing = False

    def on_middle_mouse_down(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_middle_mouse_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def pan_canvas(self, dx, dy):
        self.canvas.xview_scroll(dx, "units")
        self.canvas.yview_scroll(dy, "units")

    # ------------------- Actions -------------------
    def undo_last(self):
        if self.undo_stack:
            self.grid = self.undo_stack.pop()
            self.draw_grid()

    def restart_map(self):
        self.initialize_borders()
        self.draw_grid()

    def save_map(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if not filename:
            return
        with open(filename, "w", encoding="utf-8") as f:
            for i in range(self.grid_size):
                line = ""
                for j in range(self.grid_size):
                    t = torch.argmax(self.grid[:, i, j]).item()
                    line += inv_char_dict.get(t, ".")
                f.write(line + "\n")

    def load_map(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not filename:
            return
        with open(filename, "r", encoding="utf-8") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        n = len(lines)
        self.grid_size = n
        self.size_var.set(n)
        self.grid = torch.zeros((8, n, n))
        for i, line in enumerate(lines):
            for j, ch in enumerate(line):
                self.grid[char_dict.get(ch, GRASS), i, j] = 1
        self.update_window_size()
        self.draw_grid()


def main():
    root = tk.Tk()
    MapEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
