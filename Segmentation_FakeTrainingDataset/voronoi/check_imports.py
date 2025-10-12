import importlib.util as u

mods = [
    "tkinter","ttkbootstrap","numpy","scipy","matplotlib",
    "skimage","shapely","networkx","numba","tifffile",
    "cv2","PIL","yaml","elasticdeform","SimpleITK"
]

def ok(name, alt=None):
    spec = u.find_spec(alt or name)
    return spec is not None

for m in mods:
    alt = "PIL.Image" if m == "PIL" else None
    print(f"{m}: {'OK' if ok(m, alt) else 'FAIL'}")
