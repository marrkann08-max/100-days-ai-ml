import moderngl
import numpy as np

from .physics import DT, world_size
from .rules import MAX_TYPES

TILE = 256

STEP_SHADER = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosIn { vec2 pos_in[]; };
layout(std430, binding = 1) buffer Vel { vec2 vel[]; };
layout(std430, binding = 2) readonly buffer Type { int kind[]; };
layout(std430, binding = 3) writeonly buffer PosOut { vec2 pos_out[]; };

uniform int n;
uniform int types;
uniform float matrix[64];
uniform vec2 size;
uniform float radius;
uniform float beta;
uniform float friction;
uniform float force_scale;
uniform float dt;
uniform vec2 mouse;
uniform float mouse_force;
uniform float mouse_radius;

shared vec2 tile_pos[256];
shared int tile_kind[256];

float force(float r, float a) {
    if (r < beta) return r / beta - 1.0;
    if (r < 1.0) return a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
    return 0.0;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint local = gl_LocalInvocationID.x;
    vec2 p = i < n ? pos_in[i] : vec2(0.0);
    int ki = i < n ? kind[i] : 0;
    vec2 total = vec2(0.0);

    // Brute force, but tiled: each workgroup loads 256 particles into shared memory at a time.
    for (int base = 0; base < n; base += 256) {
        int j = base + int(local);
        tile_pos[local] = j < n ? pos_in[j] : vec2(0.0);
        tile_kind[local] = j < n ? kind[j] : 0;
        barrier();
        int count = min(256, n - base);
        for (int k = 0; k < count; k++) {
            vec2 d = tile_pos[k] - p;
            d -= round(d / size) * size;
            float r = length(d);
            if (r > 0.0 && r < radius)
                total += d / r * force(r / radius, matrix[ki * types + tile_kind[k]]);
        }
        barrier();
    }
    if (i >= n) return;

    vec2 v = vel[i] * friction + total * radius * force_scale * dt;
    if (mouse_force != 0.0) {
        vec2 dm = p - mouse;
        dm -= round(dm / size) * size;
        float rm = length(dm);
        if (rm > 0.0 && rm < mouse_radius)
            v += dm / rm * mouse_force * (1.0 - rm / mouse_radius) * dt;
    }
    vel[i] = v;
    vec2 q = p + v * dt;
    pos_out[i] = q - floor(q / size) * size;
}
"""

DRAW_VERTEX = """
#version 430
layout(std430, binding = 0) readonly buffer Pos { vec2 pos[]; };
layout(std430, binding = 2) readonly buffer Type { int kind[]; };
uniform vec2 size;
uniform float point_size;
uniform vec3 palette[8];
out vec3 color;
void main() {
    vec2 p = pos[gl_VertexID] / size * 2.0 - 1.0;
    gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
    gl_PointSize = point_size;
    color = palette[kind[gl_VertexID]];
}
"""

DRAW_FRAGMENT = """
#version 430
in vec3 color;
out vec4 frag;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float glow = exp(-r2 * 3.0);
    frag = vec4(color * glow, glow);
}
"""

FADE_VERTEX = """
#version 430
in vec2 in_pos;
out vec2 uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); uv = in_pos * 0.5 + 0.5; }
"""

FADE_FRAGMENT = """
#version 430
uniform sampler2D image;
uniform float keep;
in vec2 uv;
out vec4 frag;
void main() { frag = vec4(texture(image, uv).rgb * keep, 1.0); }
"""

PALETTE = np.array([
    (1.00, .36, .32), (.98, .78, .28), (.40, .92, .52), (.25, .78, 1.0),
    (.62, .48, 1.0), (1.0, .45, .82), (.95, .95, .9), (.35, .95, .9),
], "f4")


class World:
    """A particle-life world living entirely in GPU buffers."""

    def __init__(self, ctx, rules, particles=8000, seed=None, aspect=16 / 9):
        self.ctx = ctx
        self.n = particles
        self.size = world_size(particles, aspect)
        self.rng = np.random.default_rng(seed)
        self.step_program = ctx.compute_shader(STEP_SHADER)
        self.draw_program = ctx.program(vertex_shader=DRAW_VERTEX, fragment_shader=DRAW_FRAGMENT)
        self.draw_program["palette"].write(PALETTE.tobytes())
        self.draw_vao = ctx.vertex_array(self.draw_program, [])
        self.buffers = [ctx.buffer(reserve=particles * 8) for _ in range(2)]
        self.velocity = ctx.buffer(reserve=particles * 8)
        self.kind = ctx.buffer(reserve=particles * 4)
        self.current = 0
        self.time = 0
        self.mouse = (0.0, 0.0, 0.0, .15)
        self.set_rules(rules, reshuffle=True)

    def set_rules(self, rules, reshuffle=False):
        old_types = getattr(self, "rules", None)
        self.rules = rules
        if reshuffle or old_types is None or old_types.types != rules.types:
            self.scatter()
        m = np.zeros(MAX_TYPES * MAX_TYPES, "f4")
        m[:rules.types ** 2] = rules.matrix.ravel()
        p = self.step_program
        p["matrix"].write(m.tobytes())
        p["types"] = rules.types
        p["radius"] = rules.radius
        p["beta"] = rules.beta
        p["friction"] = rules.friction(DT)
        p["force_scale"] = rules.force

    def scatter(self):
        pos = (self.rng.random((self.n, 2)) * self.size).astype("f4")
        kinds = self.rng.integers(0, self.rules.types, self.n).astype("i4")
        self.load(pos, np.zeros_like(pos), kinds)

    def load(self, pos, vel, kinds):
        self.buffers[self.current].write(np.asarray(pos, "f4").tobytes())
        self.velocity.write(np.asarray(vel, "f4").tobytes())
        self.kind.write(np.asarray(kinds, "i4").tobytes())
        self.time = 0

    def stir(self, x, y, strength, radius=.15):
        """Push (positive) or pull (negative) particles around a world-space point."""
        self.mouse = (x, y, strength, radius)

    def step(self, steps=1):
        p = self.step_program
        p["n"] = self.n
        p["size"] = tuple(self.size)
        p["dt"] = DT
        x, y, strength, radius = self.mouse
        p["mouse"] = (x, y)
        p["mouse_force"] = strength
        p["mouse_radius"] = radius
        groups = (self.n + TILE - 1) // TILE
        self.velocity.bind_to_storage_buffer(1)
        self.kind.bind_to_storage_buffer(2)
        for _ in range(steps):
            self.buffers[self.current].bind_to_storage_buffer(0)
            self.buffers[1 - self.current].bind_to_storage_buffer(3)
            p.run(groups)
            self.current = 1 - self.current
        self.time += steps

    def positions(self):
        return np.frombuffer(self.buffers[self.current].read(), "f4").reshape(-1, 2).copy()

    def velocities(self):
        return np.frombuffer(self.velocity.read(), "f4").reshape(-1, 2).copy()

    def kinds(self):
        return np.frombuffer(self.kind.read(), "i4").copy()

    def draw(self, point_size=3.0):
        self.buffers[self.current].bind_to_storage_buffer(0)
        self.kind.bind_to_storage_buffer(2)
        self.draw_program["size"] = tuple(self.size)
        self.draw_program["point_size"] = point_size
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.draw_vao.render(moderngl.POINTS, vertices=self.n)

    def release(self):
        for item in (*self.buffers, self.velocity, self.kind, self.draw_vao, self.draw_program, self.step_program):
            item.release()


class Trails:
    """Draws particles into a texture that fades a little each frame, leaving glowing trails."""

    def __init__(self, ctx, size):
        self.ctx = ctx
        self.textures = [ctx.texture(size, 3) for _ in range(2)]
        self.fbos = [ctx.framebuffer(color_attachments=[t]) for t in self.textures]
        for fbo in self.fbos:
            fbo.clear()
        self.program = ctx.program(vertex_shader=FADE_VERTEX, fragment_shader=FADE_FRAGMENT)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4")
        self.buffer = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "2f", "in_pos")])
        self.current = 0

    def render(self, world, point_size, keep):
        """Fade the previous frame into the other buffer, add this frame's particles, return the texture."""
        src, dst = self.current, 1 - self.current
        self.fbos[dst].use()
        self.ctx.disable(moderngl.BLEND)
        self.textures[src].use(0)
        self.program["keep"] = keep
        self.vao.render(moderngl.TRIANGLE_STRIP)
        world.draw(point_size)
        self.current = dst
        return self.textures[dst]

    def present(self, texture):
        """Copy a trails texture to whatever framebuffer is bound."""
        self.ctx.disable(moderngl.BLEND)
        texture.use(0)
        self.program["keep"] = 1.0
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def clear(self):
        for fbo in self.fbos:
            fbo.clear()

    def release(self):
        for item in (*self.fbos, *self.textures, self.vao, self.buffer, self.program):
            item.release()
