#!/usr/bin/env python3
"""Build a clean MG996R-driven two-finger gripper in SolidWorks 2025."""

import math
import os
import time

import pythoncom
import win32com.client


ROOT = r"C:\Users\hanni\Desktop\gripper_v2"
PART_TEMPLATE = (
    r"C:\ProgramData\SOLIDWORKS\SOLIDWORKS 2025\templates\gb_part.prtdot"
)
ASSEMBLY_TEMPLATE = (
    r"C:\ProgramData\SOLIDWORKS\SOLIDWORKS 2025\templates\gb_assembly.asmdot"
)

MM = 0.001
NULL = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)


def connect():
    try:
        app = win32com.client.GetActiveObject("SldWorks.Application")
    except Exception:
        app = win32com.client.Dispatch("SldWorks.Application")
    app.Visible = True
    return app


def wait_for_active(app, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = app.ActiveDoc
        if doc is not None:
            return doc
        time.sleep(0.2)
    raise RuntimeError("SolidWorks did not create an active document")


def new_doc(app, template):
    result = app.NewDocument(template, 0, 0.001, 0.001)
    if result is None:
        raise RuntimeError(f"Could not create document from {template}")
    return wait_for_active(app)


def select_front_plane(doc):
    doc.ClearSelection2(True)
    ok = doc.Extension.SelectByID2(
        "前视基准面", "PLANE", 0, 0, 0, False, 0, NULL, 0
    )
    if not ok:
        raise RuntimeError("Could not select 前视基准面")


def begin_sketch(doc):
    select_front_plane(doc)
    doc.SketchManager.InsertSketch(True)
    sketch = doc.SketchManager.ActiveSketch
    if sketch is None:
        raise RuntimeError("Could not start sketch")
    doc.SketchManager.AddToDB = True
    doc.SketchManager.DisplayWhenAdded = False
    return sketch


def line(doc, a, b):
    doc.SketchManager.CreateLine(
        a[0] * MM, a[1] * MM, 0, b[0] * MM, b[1] * MM, 0
    )


def polygon(doc, points):
    for index, point in enumerate(points):
        line(doc, point, points[(index + 1) % len(points)])


def rectangle(doc, x1, y1, x2, y2):
    doc.SketchManager.CreateCornerRectangle(
        x1 * MM, y1 * MM, 0, x2 * MM, y2 * MM, 0
    )


def circle(doc, x, y, radius):
    doc.SketchManager.CreateCircleByRadius(x * MM, y * MM, 0, radius * MM)


def finish_sketch(doc):
    sketch = doc.SketchManager.ActiveSketch
    doc.SketchManager.AddToDB = False
    doc.SketchManager.DisplayWhenAdded = True
    doc.SketchManager.InsertSketch(True)
    return sketch


def hide_feature_sketch(doc, feature):
    if feature is None:
        return
    try:
        sketch_feature = feature.GetFirstSubFeature()
    except TypeError:
        try:
            sketch_feature = feature.GetFirstSubFeature
        except Exception:
            sketch_feature = None
    except Exception:
        sketch_feature = None
    while sketch_feature is not None:
        doc.ClearSelection2(True)
        try:
            sketch_feature.Select2(False, 0)
            doc.BlankSketch()
        except Exception:
            pass
        try:
            sketch_feature = sketch_feature.GetNextSubFeature()
        except TypeError:
            sketch_feature = sketch_feature.GetNextSubFeature
        except AttributeError:
            break
    doc.ClearSelection2(True)


def extrude(doc, depth_mm, name):
    depth = depth_mm * MM
    feature = doc.FeatureManager.FeatureExtrusion2(
        True,
        False,
        False,
        0,
        0,
        depth,
        depth,
        False,
        False,
        False,
        False,
        0,
        0,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        0,
        0,
        False,
    )
    if feature is None:
        raise RuntimeError(f"Extrusion failed: {name}")
    try:
        feature.Name = name
    except Exception:
        pass
    return feature


def sketch_extrude(doc, depth_mm, name, draw):
    begin_sketch(doc)
    draw(doc)
    finish_sketch(doc)
    feature = extrude(doc, depth_mm, name)
    hide_feature_sketch(doc, feature)
    _ = doc.EditRebuild3
    return feature


def set_document_color(doc, rgb):
    # SolidWorks material values: RGB, ambient, diffuse, specular,
    # shininess, transparency, emission.
    values = (
        rgb[0],
        rgb[1],
        rgb[2],
        0.30,
        0.75,
        0.20,
        0.25,
        0.0,
        0.0,
    )
    try:
        doc.MaterialPropertyValues = win32com.client.VARIANT(
            pythoncom.VT_ARRAY | pythoncom.VT_R8, values
        )
    except Exception:
        pass


def hide_all_sketches(doc):
    features = doc.FeatureManager.GetFeatures(False)
    for feature in features or []:
        try:
            feature_type = feature.GetTypeName2
            if feature_type in ("ProfileFeature", "3DProfileFeature"):
                doc.ClearSelection2(True)
                feature.Select2(False, 0)
                doc.BlankSketch()
        except Exception:
            pass
    doc.ClearSelection2(True)


def save_part(app, doc, filename, color):
    set_document_color(doc, color)
    hide_all_sketches(doc)
    _ = doc.EditRebuild3
    path = os.path.join(ROOT, filename)
    doc.SaveAs3(path, 0, 1)
    time.sleep(0.2)
    if not os.path.exists(path):
        raise RuntimeError(f"Could not save {path}")
    print(f"OK  {filename}")
    try:
        title = doc.GetTitle
        if callable(title):
            title = title()
        app.CloseDoc(title)
    except Exception:
        pass
    return path


def make_part(app, filename, color, builder):
    doc = new_doc(app, PART_TEMPLATE)
    builder(doc)
    return save_part(app, doc, filename, color)


def capsule_points(a, b, radius, segments=8):
    """Return a closed chamfered-link outline around endpoints A and B."""
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / length, (by - ay) / length
    px, py = -uy, ux

    def point(origin, along, across):
        return (
            origin[0] + ux * along + px * across,
            origin[1] + uy * along + py * across,
        )

    return [
        point(a, -0.55 * radius, radius),
        point(b, 0.55 * radius, radius),
        point(b, radius, 0.55 * radius),
        point(b, radius, -0.55 * radius),
        point(b, 0.55 * radius, -radius),
        point(a, -0.55 * radius, -radius),
        point(a, -radius, -0.55 * radius),
        point(a, -radius, 0.55 * radius),
    ]


def build_frame(doc):
    outer = [
        (-45, -35),
        (45, -35),
        (50, -30),
        (50, 24),
        (44, 30),
        (-44, 30),
        (-50, 24),
        (-50, -30),
    ]

    def draw(d):
        polygon(d, outer)
        rectangle(d, -30, -28, 30, -14)
        rectangle(d, -30, 3, 30, 10)
        circle(d, 0, -5.5, 6.7)
        for x in (-38, 38):
            circle(d, x, 16, 4.1)
        for x in (-40, 40):
            circle(d, x, -25, 2.1)
        for x in (-24, 24):
            circle(d, x, -8, 1.7)

    sketch_extrude(doc, 5, "Frame_5mm", draw)


def build_servo(doc):
    # The mounting silhouette follows the nominal MG996R envelope.
    body = [
        (-20.35, -20),
        (20.35, -20),
        (20.35, 7),
        (28, 7),
        (28, 13),
        (20.35, 13),
        (20.35, 20),
        (-20.35, 20),
        (-20.35, 13),
        (-28, 13),
        (-28, 7),
        (-20.35, 7),
    ]

    def main(d):
        polygon(d, body)
        circle(d, -24, 10, 1.7)
        circle(d, 24, 10, 1.7)

    sketch_extrude(doc, 19.7, "MG996R_Case", main)
    sketch_extrude(
        doc,
        22.0,
        "Top_Cap",
        lambda d: rectangle(d, -20.35, 13, 20.35, 20),
    )
    sketch_extrude(doc, 24.0, "Output_Boss", lambda d: circle(d, 0, 12.5, 6.2))
    sketch_extrude(doc, 28.0, "Output_Spline", lambda d: circle(d, 0, 12.5, 2.9))

    def screw_heads(d):
        for x in (-16, 16):
            for y in (-16, 16):
                circle(d, x, y, 1.4)

    sketch_extrude(doc, 20.5, "Case_Screws", screw_heads)


def build_horn(doc):
    outline = [
        (-22, -4),
        (22, -4),
        (25, 0),
        (22, 4),
        (-22, 4),
        (-25, 0),
    ]

    def arm(d):
        polygon(d, outline)
        circle(d, 0, 0, 2.9)
        circle(d, -20, 0, 1.7)
        circle(d, 20, 0, 1.7)
        for x in (-14, -8, 8, 14):
            circle(d, x, 0, 1.0)

    sketch_extrude(doc, 3, "Double_Servo_Horn", arm)
    sketch_extrude(doc, 5, "Horn_Hub", lambda d: circle(d, 0, 0, 5.4))


LEFT_JAW = [
    (-14, -14),
    (16, -14),
    (17, 4),
    (14, 18),
    (10, 38),
    (8, 58),
    (20, 73),
    (19, 84),
    (12, 92),
    (4, 80),
    (-4, 70),
    (-7, 55),
    (-10, 35),
    (-13, 15),
]


def build_jaw(doc, mirrored=False):
    points = [(-x, y) for x, y in LEFT_JAW] if mirrored else LEFT_JAW
    link_x = -12 if mirrored else 12

    def draw(d):
        polygon(d, points)
        circle(d, 0, 0, 4.1)
        circle(d, link_x, -3, 2.1)

    sketch_extrude(doc, 8, "Right_Jaw" if mirrored else "Left_Jaw", draw)


def build_link(doc, mirrored=False):
    if mirrored:
        a, b = (20, -5.5), (26, 13)
    else:
        a, b = (-20, -5.5), (-26, 13)
    outline = capsule_points(a, b, 4.0)

    def draw(d):
        polygon(d, outline)
        circle(d, a[0], a[1], 1.7)
        circle(d, b[0], b[1], 2.1)

    sketch_extrude(doc, 4, "Right_Link" if mirrored else "Left_Link", draw)


def build_pad(doc, mirrored=False):
    points = [(10, 68), (21, 72), (20, 84), (9, 80)]
    if mirrored:
        points = [(-x, y) for x, y in points]

    sketch_extrude(
        doc,
        4,
        "Right_Rubber_Pad" if mirrored else "Left_Rubber_Pad",
        lambda d: polygon(d, points),
    )

    # Four raised ribs make the contact surface visually and functionally clear.
    x1, x2 = (10.5, 20.5) if not mirrored else (-20.5, -10.5)

    def ribs(d):
        for y in (72, 75, 78, 81):
            rectangle(d, x1, y, x2, y + 1.2)

    sketch_extrude(doc, 5, "Grip_Ribs", ribs)


def build_pin(doc, radius, depth, name):
    sketch_extrude(doc, depth, name, lambda d: circle(d, 0, 0, radius))


def component_color(component, rgb):
    values = (
        rgb[0],
        rgb[1],
        rgb[2],
        0.30,
        0.75,
        0.25,
        0.30,
        0.0,
        0.0,
    )
    try:
        component.MaterialPropertyValues = win32com.client.VARIANT(
            pythoncom.VT_ARRAY | pythoncom.VT_R8, values
        )
    except Exception:
        pass


def add_component(assembly, path, xyz_mm, color=None):
    x, y, z = (value * MM for value in xyz_mm)
    component = assembly.AddComponent5(path, 0, 0, False, "", x, y, z)
    if component is None:
        raise RuntimeError(f"Could not insert {path}")
    if color is not None:
        component_color(component, color)
    return component


def build_assembly(app, paths):
    for path in dict.fromkeys(paths.values()):
        if app.OpenDoc(path, 1) is None:
            raise RuntimeError(f"Could not preload {path}")
    assembly = new_doc(app, ASSEMBLY_TEMPLATE)
    components = []

    def add(name, xyz, color=None):
        component = add_component(assembly, paths[name], xyz, color)
        components.append(component)
        return component

    frame = add("frame", (0, 0, 0), (0.15, 0.28, 0.48))
    add("servo", (0, -18, -19.7), (0.12, 0.12, 0.15))
    add("horn", (0, -5.5, 8), (0.88, 0.88, 0.90))
    add("left_jaw", (-38, 16, 6), (0.92, 0.48, 0.08))
    add("right_jaw", (38, 16, 6), (0.92, 0.48, 0.08))
    add("left_link", (0, 0, 14), (0.82, 0.84, 0.88))
    add("right_link", (0, 0, 14), (0.82, 0.84, 0.88))
    add("left_pad", (-38, 16, 12), (0.08, 0.08, 0.08))
    add("right_pad", (38, 16, 12), (0.08, 0.08, 0.08))

    for x in (-38, 38):
        add("pivot_pin", (x, 16, -1), (0.72, 0.74, 0.78))
    for x, y in ((-20, -5.5), (20, -5.5), (-26, 13), (26, 13)):
        add("link_pin", (x, y, 7), (0.72, 0.74, 0.78))

    # Fix components at their nominal open-position coordinates. The geometry
    # remains split into real parts and can receive motion mates later.
    for component in components:
        assembly.ClearSelection2(True)
        try:
            component.Select4(False, NULL, False)
            _ = assembly.FixComponent
        except Exception:
            pass

    assembly.ClearSelection2(True)
    _ = assembly.EditRebuild3
    try:
        assembly.ShowNamedView2("*Isometric", 7)
        _ = assembly.ViewZoomtofit2
        _ = assembly.ViewDisplayShadedWithEdges
    except Exception:
        pass

    assembly_path = os.path.join(ROOT, "MG996R_parallel_gripper.SLDASM")
    assembly.SaveAs3(assembly_path, 0, 1)
    time.sleep(0.2)
    if not os.path.exists(assembly_path):
        raise RuntimeError("Could not save assembly")

    preview_path = os.path.join(ROOT, "MG996R_parallel_gripper_preview.bmp")
    try:
        assembly.SaveBMP(preview_path, 1600, 1000)
    except Exception as exc:
        print(f"WARN preview not saved: {exc}")

    print(f"OK  {os.path.basename(assembly_path)}")
    return assembly, assembly_path, preview_path


def main():
    pythoncom.CoInitialize()
    os.makedirs(ROOT, exist_ok=True)
    app = connect()
    try:
        app.CloseAllDocuments(True)
    except Exception:
        pass

    colors = {
        "frame": (0.15, 0.28, 0.48),
        "servo": (0.12, 0.12, 0.15),
        "metal": (0.82, 0.84, 0.88),
        "jaw": (0.92, 0.48, 0.08),
        "rubber": (0.08, 0.08, 0.08),
    }

    print("Building MG996R parallel gripper...")
    paths = {}
    paths["frame"] = make_part(
        app, "01_frame.SLDPRT", colors["frame"], build_frame
    )
    paths["servo"] = make_part(
        app, "02_MG996R_servo.SLDPRT", colors["servo"], build_servo
    )
    paths["horn"] = make_part(
        app, "03_double_servo_horn.SLDPRT", colors["metal"], build_horn
    )
    paths["left_jaw"] = make_part(
        app,
        "04_left_jaw.SLDPRT",
        colors["jaw"],
        lambda doc: build_jaw(doc, False),
    )
    paths["right_jaw"] = make_part(
        app,
        "05_right_jaw.SLDPRT",
        colors["jaw"],
        lambda doc: build_jaw(doc, True),
    )
    paths["left_link"] = make_part(
        app,
        "06_left_link.SLDPRT",
        colors["metal"],
        lambda doc: build_link(doc, False),
    )
    paths["right_link"] = make_part(
        app,
        "07_right_link.SLDPRT",
        colors["metal"],
        lambda doc: build_link(doc, True),
    )
    paths["left_pad"] = make_part(
        app,
        "08_left_grip_pad.SLDPRT",
        colors["rubber"],
        lambda doc: build_pad(doc, False),
    )
    paths["right_pad"] = make_part(
        app,
        "09_right_grip_pad.SLDPRT",
        colors["rubber"],
        lambda doc: build_pad(doc, True),
    )
    paths["pivot_pin"] = make_part(
        app,
        "10_pivot_pin.SLDPRT",
        colors["metal"],
        lambda doc: build_pin(doc, 4.0, 19, "Pivot_Pin"),
    )
    paths["link_pin"] = make_part(
        app,
        "11_link_pin.SLDPRT",
        colors["metal"],
        lambda doc: build_pin(doc, 1.65, 12, "Link_Pin"),
    )

    _, assembly_path, preview_path = build_assembly(app, paths)
    print(f"Assembly: {assembly_path}")
    print(f"Preview:  {preview_path}")
    print("DONE")


if __name__ == "__main__":
    main()
