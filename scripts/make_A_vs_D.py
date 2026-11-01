"""Build A vs D_distill_clean side-by-side GIF for the report locally."""
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
a_path = HERE / 'tail_round1/A_bigkick_realtime.gif'   # 100 frames @80ms = 8s
d_path = HERE / 'tail_round2/D_distill_clean_bigkick.gif'  # 200 frames @40ms = 8s

a_frames = []
im = Image.open(a_path)
for i in range(im.n_frames):
    im.seek(i)
    a_frames.append(im.convert('RGB').copy())

d_frames = []
im = Image.open(d_path)
for i in range(im.n_frames):
    im.seek(i)
    d_frames.append(im.convert('RGB').copy())

out = []
for i, frame in enumerate(d_frames):
    ai = min(len(a_frames) - 1, int(i * len(a_frames) / len(d_frames)))
    a = a_frames[ai].resize(frame.size)
    w, h = frame.size
    canvas = Image.new('RGB', (2 * w, h + 30), (255, 255, 255))
    canvas.paste(a, (0, 30))
    canvas.paste(frame, (w, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'A: original policy, frozen last frame (falls at 2.78s)', fill='black')
    draw.text((w + 8, 8), 'D: distilled policy + tail (stands 8s)', fill='black')
    out.append(canvas)

out[0].save(HERE / 'A_vs_D_bigkick.gif', save_all=True, append_images=out[1:], duration=40, loop=0)
print('saved A_vs_D_bigkick.gif', len(out), 'frames')
