"""Render a real textured glTF mesh with a moving perspective camera and studio lighting."""

import argparse
import math
from pathlib import Path
import subprocess

import moderngl
import numpy as np
from PIL import Image, ImageDraw
import trimesh

VERTEX = """
#version 330
uniform mat4 mvp;
in vec3 pos; in vec3 normal; in vec2 uv;
out vec3 vpos; out vec3 norm; out vec2 tex;
void main(){vpos=pos; norm=normal; tex=uv; gl_Position=mvp*vec4(pos,1);}
"""
FRAGMENT = """
#version 330
uniform float photoWeight; uniform sampler2D pigment; uniform vec3 eye; uniform vec4 factor;
in vec3 vpos; in vec3 norm; in vec2 tex; out vec4 color;
float spec(vec3 n,vec3 v,vec3 l,float tight){return pow(max(dot(reflect(-l,n),v),0.),tight);}
void main(){
  vec4 sampleColor=texture(pigment,tex)*factor;
  if(sampleColor.a<.12) discard;
  vec3 n=normalize(norm), v=normalize(eye-vpos);
  if(dot(n,v)<0.) n=-n;
  vec3 key=normalize(vec3(-.70,.85,1.0)), edge=normalize(vec3(.85,.40,-.50));
  float soft=.66+.48*max(dot(n,key),0.)+.23*max(dot(n,edge),0.);
  vec3 base=sampleColor.rgb*soft;
  base+=vec3(.10,.13,.16)*spec(n,v,key,17.)+vec3(.08,.11,.15)*spec(n,v,edge,9.);
  color=vec4(clamp(mix(base,sampleColor.rgb,photoWeight),0.,1.),1.);
}
"""


def unit(vector):
    return vector / np.linalg.norm(vector)


class Stage:
    def __init__(self, asset, size):
        self.size = size
        self.ctx = moderngl.create_standalone_context()
        self.fbo = self.ctx.simple_framebuffer(size, components=3)
        self.fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        scene = trimesh.load_scene(asset, process=False)
        meshes = list(scene.dump())
        self.vertices = np.vstack([m.vertices for m in meshes])
        center = (self.vertices.min(axis=0) + self.vertices.max(axis=0)) / 2
        scale = np.ptp(self.vertices, axis=0).max()
        self.vertices = (self.vertices - center) / scale
        self.parts = []
        for mesh in meshes:
            mat = mesh.visual.material
            img = getattr(mat, "baseColorTexture", None)
            if img is None:
                img = Image.new("RGBA", (1, 1), "white")
            img = img.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            tex = self.ctx.texture(img.size, 4, img.tobytes())
            tex.build_mipmaps()
            uv = mesh.visual.uv
            if uv is None:
                uv = np.zeros((len(mesh.vertices), 2))
            data = np.column_stack(
                ((mesh.vertices - center) / scale, mesh.vertex_normals, uv)
            ).astype("f4")
            vertices = self.ctx.buffer(data.tobytes())
            indices = self.ctx.buffer(mesh.faces.astype("u4").tobytes())
            geom = self.ctx.vertex_array(
                self.program, [(vertices, "3f 3f 2f", "pos", "normal", "uv")], indices
            )
            factor = getattr(mat, "baseColorFactor", None)
            factor = (
                tuple(np.asarray(factor) / 255.0)
                if factor is not None
                else (1.0, 1.0, 1.0, 1.0)
            )
            self.parts.append(
                (geom, tex, factor, 0.96 if mat.name == "CalibratedReference" else 0.0)
            )

    def frame(self, angle, elevation, roll, fill, focus, offset):
        w, h = self.size
        az, el, bank = np.deg2rad([angle, elevation, roll])
        direction = np.array(
            [math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)]
        )
        right = unit(np.cross([0, 1, 0], direction))
        up = np.cross(direction, right)
        right, up = (
            right * np.cos(bank) + up * np.sin(bank),
            up * np.cos(bank) - right * np.sin(bank),
        )
        # Fit perspective-projected corners, then move into the selected detail.
        p = self.vertices[::24]
        f = 1 / math.tan(math.radians(16))
        dist = max(
            np.max(abs(p @ right) * f * h / w / fill + p @ direction),
            np.max(abs(p @ up) * f / fill + p @ direction),
        )
        target = right * focus[0] + up * focus[1]
        eye = target + direction * dist
        view = np.eye(4)
        view[:3, :3] = np.stack((right, up, direction))
        view[:3, 3] = -(view[:3, :3] @ eye)
        proj = np.zeros((4, 4))
        proj[0, 0], proj[1, 1], proj[1, 2] = f * h / w, f, -offset
        proj[2, 2], proj[2, 3], proj[3, 2] = -1.002, -0.02002, -1
        self.mvp = proj @ view
        self.program["mvp"].write(self.mvp.T.astype("f4").tobytes())
        self.program["eye"].value = tuple(eye)
        self.fbo.clear(0.032, 0.039, 0.050, 1.0, depth=1.0)
        for geom, tex, factor, weight in self.parts:
            self.program["photoWeight"].value = weight
            self.program["factor"].value = factor
            tex.use(0)
            geom.render()
        return Image.frombytes(
            "RGB", self.size, self.fbo.read(components=3, alignment=1)
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def render(
    asset,
    path,
    *,
    size=(1080, 1920),
    fps=30,
    seconds=5,
    angles=(-18, 16),
    elevation=8.0,
    roll=-12.0,
    fill=0.86,
    focus=(0.0, 0.0),
    offset=0.08,
):
    stage = Stage(asset, size)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size[0]}x{size[1]}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "16",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(round(seconds * fps)):
            t = i / max(1, round(seconds * fps) - 1)
            angle = angles[0] + (angles[1] - angles[0]) * t
            im = stage.frame(
                angle, elevation + 2 * t, roll + 3 * t, fill, focus, offset
            )
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Product stage encode failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def contact(asset, output):
    size = (210, 290)
    stage = Stage(asset, size)
    page = Image.new("RGB", (size[0] * 4, size[1] * 2))
    for n in range(8):
        im = stage.frame(n * 45, 8, 0, 0.87, (0.0, 0.0), 0)
        ImageDraw.Draw(im).text((6, 6), str(n * 45), fill="white")
        page.paste(im, ((n % 4) * size[0], (n // 4) * size[1]))
    page.save(output)


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shots", type=Path)
    args = parser.parse_args()
    if args.shots:
        recipes = json.loads(args.shots.read_text(encoding="utf-8"))
        for name, recipe in recipes.items():
            render(args.asset, args.output / name, **recipe)
    else:
        contact(args.asset, args.output)
