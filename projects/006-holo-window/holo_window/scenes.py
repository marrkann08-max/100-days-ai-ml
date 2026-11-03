import math

import numpy as np

from .depth import photo_mesh

BACKGROUND = (.035, .045, .055)

COLOR_VERTEX = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_color;
out vec3 color;
out float depth;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    color = in_color;
    depth = -in_pos.z;
}
"""

COLOR_FRAGMENT = """
#version 330
uniform vec3 background;
uniform float fog_depth;
in vec3 color;
in float depth;
out vec4 frag;
void main() {
    float fog = clamp(depth / fog_depth, 0.0, 0.65);
    frag = vec4(mix(color, background, fog), 1.0);
}
"""

PHOTO_VERTEX = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    uv = in_uv;
}
"""

PHOTO_FRAGMENT = """
#version 330
uniform sampler2D photo;
in vec2 uv;
out vec4 frag;
void main() {
    frag = vec4(texture(photo, uv).rgb, 1.0);
}
"""


class Mesh:
    """Coloured triangles collected in numpy before they go to the GPU."""

    def __init__(self):
        self.parts = []

    def triangles(self, points, color):
        points = np.asarray(points, np.float32).reshape(-1, 3)
        colors = np.broadcast_to(np.asarray(color, np.float32), points.shape)
        self.parts.append(np.hstack((points, colors)))

    def quad(self, a, b, c, d, color):
        self.triangles([a, b, c, a, c, d], color)

    def disc(self, center, radius, color, normal_axis=2, segments=40, flatten=1.0):
        """A filled circle facing along one axis (2 = towards the viewer, 1 = up)."""
        angles = np.linspace(0, 2 * math.pi, segments + 1)
        ring = np.zeros((segments + 1, 3))
        axes = [i for i in range(3) if i != normal_axis]
        ring[:, axes[0]] = np.cos(angles) * radius
        ring[:, axes[1]] = np.sin(angles) * radius * flatten
        ring += center
        fan = np.stack((np.repeat([center], segments, 0), ring[:-1], ring[1:]), axis=1)
        self.triangles(fan, color)

    def array(self):
        return np.vstack(self.parts).astype(np.float32)


def grid_lines(mesh, origin, u, v, length_u, length_v, spacing, width, color):
    """Thin ribbons lying in the plane spanned by unit vectors u and v."""
    origin, u, v = (np.asarray(x, float) for x in (origin, u, v))
    for i in range(int(round(length_u / spacing)) + 1):
        a = origin + u * min(i * spacing, length_u)
        mesh.quad(a - u * width / 2, a + u * width / 2, a + u * width / 2 + v * length_v, a - u * width / 2 + v * length_v, color)
    for j in range(int(round(length_v / spacing)) + 1):
        a = origin + v * min(j * spacing, length_v)
        mesh.quad(a - v * width / 2, a - v * width / 2 + u * length_u, a + v * width / 2 + u * length_u, a + v * width / 2, color)


TARGETS = (  # x, y as fractions of the half-window; z in cm (positive pops out of the screen)
    (-.55, .35, -38, 2.4), (.5, .45, -30, 2.2), (.1, -.25, -18, 2.0),
    (-.35, -.4, -9, 1.8), (.6, -.2, -4, 1.6), (-.05, .15, 6, 1.4), (.28, .3, 11, 1.1),
)


def room_mesh(width, height, depth=50.0):
    """A box sunk into the screen with grid walls and floating bullseye targets on stalks."""
    mesh = Mesh()
    hw, hh = width / 2, height / 2
    walls = (.07, .09, .11)
    back = (.05, .065, .08)
    line = (.25, .75, .85)
    lift = .03
    # Walls: floor, ceiling, left, right and back, then grids just in front of them.
    mesh.quad((-hw, -hh, 0), (hw, -hh, 0), (hw, -hh, -depth), (-hw, -hh, -depth), walls)
    mesh.quad((-hw, hh, 0), (-hw, hh, -depth), (hw, hh, -depth), (hw, hh, 0), walls)
    mesh.quad((-hw, -hh, 0), (-hw, -hh, -depth), (-hw, hh, -depth), (-hw, hh, 0), walls)
    mesh.quad((hw, -hh, 0), (hw, hh, 0), (hw, hh, -depth), (hw, -hh, -depth), walls)
    mesh.quad((-hw, -hh, -depth), (hw, -hh, -depth), (hw, hh, -depth), (-hw, hh, -depth), back)
    spacing, w = 5.0, .12
    grid_lines(mesh, (-hw, -hh + lift, 0), (1, 0, 0), (0, 0, -1), width, depth, spacing, w, line)
    grid_lines(mesh, (-hw, hh - lift, 0), (1, 0, 0), (0, 0, -1), width, depth, spacing, w, line)
    grid_lines(mesh, (-hw + lift, -hh, 0), (0, 1, 0), (0, 0, -1), height, depth, spacing, w, line)
    grid_lines(mesh, (hw - lift, -hh, 0), (0, 1, 0), (0, 0, -1), height, depth, spacing, w, line)
    grid_lines(mesh, (-hw, -hh, -depth + lift), (1, 0, 0), (0, 1, 0), width, height, spacing, w, line)

    for fx, fy, z, r in TARGETS:
        x, y = fx * hw, fy * hh
        stalk = (.75, .78, .8)
        for a, b in (((-.06, 0), (.06, 0)), ((0, -.06), (0, .06))):
            p, q = np.array([x + a[0], y + a[1], z]), np.array([x + b[0], y + b[1], z])
            mesh.quad(p, q, q + (0, 0, -(depth + z) + lift), p + (0, 0, -(depth + z) + lift), stalk)
        mesh.disc((x, -hh + lift * 2, z), r * .9, (.02, .025, .03), normal_axis=1, flatten=.6)
        for k, color in enumerate(((.95, .95, .95), (.85, .12, .14), (.95, .95, .95), (.85, .12, .14))):
            mesh.disc((x, y, z + .01 * (k + 1)), r * (1 - k * .24), color)
    return mesh.array()


class Room:
    name = "Box room"

    def __init__(self, ctx, settings):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=COLOR_VERTEX, fragment_shader=COLOR_FRAGMENT)
        self.program["background"].value = BACKGROUND
        self.program["fog_depth"].value = 90.0
        data = room_mesh(settings.screen_width, settings.screen_height)
        self.buffer = ctx.buffer(data.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "3f 3f", "in_pos", "in_color")])

    def draw(self, mvp):
        self.program["mvp"].write(mvp.T.astype("f4").tobytes())
        self.vao.render()


class Photo:
    """A photo turned into a depth mesh behind the glass."""

    def __init__(self, ctx, settings, name, rgb, disparity):
        self.ctx, self.name = ctx, name
        self.program = ctx.program(vertex_shader=PHOTO_VERTEX, fragment_shader=PHOTO_FRAGMENT)
        h, w = rgb.shape[:2]
        self.texture = ctx.texture((w, h), 3, np.ascontiguousarray(rgb).tobytes())
        self.texture.build_mipmaps()
        self.texture.repeat_x = self.texture.repeat_y = False
        self.parts = []
        for backdrop in (True, False):
            vertices, triangles = photo_mesh(disparity, w / h, settings, backdrop=backdrop, columns=64 if backdrop else 256)
            buffer, index = ctx.buffer(vertices.tobytes()), ctx.buffer(triangles.tobytes())
            vao = ctx.vertex_array(self.program, [(buffer, "3f 2f", "in_pos", "in_uv")], index)
            self.parts.append((vao, buffer, index))

    def draw(self, mvp):
        self.program["mvp"].write(mvp.T.astype("f4").tobytes())
        self.texture.use(0)
        for vao, _, _ in self.parts:
            vao.render()

    def release(self):
        for part in self.parts:
            for item in part:
                item.release()
        self.texture.release()
        self.program.release()
