# MG996R Parallel Gripper

This folder contains a SolidWorks 2025 design for a servo-driven two-finger
gripper. The assembly is saved in its nominal open position.

## Mechanism

- MG996R-size servo mounted behind a rigid front frame
- Double-arm servo horn converts servo rotation into link motion
- Independent left and right links drive two pivoting hook jaws
- Replaceable ribbed grip pads provide compliant contact surfaces
- Separate pivot and link pins make the load path visible in the assembly

## Nominal Dimensions

- Frame: 100 x 65 x 5 mm
- Servo envelope: approximately 40.7 x 40 x 19.7 mm, plus mounting ears
- Jaw thickness: 8 mm
- Pivot pin: 8 mm diameter
- Link pin: 3.3 mm diameter
- Initial tip opening: approximately 36 mm
- Assembly envelope: approximately 100 x 143 x 38 mm

## Files

- `MG996R_parallel_gripper.SLDASM`: main SolidWorks assembly
- `01_frame.SLDPRT` through `11_link_pin.SLDPRT`: source parts
- `MG996R_parallel_gripper.STEP`: neutral-format assembly export
- `MG996R_parallel_gripper_final.png`: isometric preview
- `MG996R_parallel_gripper_front.png`: front preview

All sketches are absorbed into features and hidden. Every source part contains
one solid body. Component positions represent the open gripper configuration.
