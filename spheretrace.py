import taichi as ti
import numpy as np
import math

ti.init(arch=[ti.opengl, ti.metal])

nx = 600
ny = 600
pixels = ti.Vector(3, dt=ti.f32, shape=(nx, ny))

lookfrom = ti.Vector(3, dt=ti.f32, shape=())
lookat = ti.Vector(3, dt=ti.f32, shape=())
up = ti.Vector(3, dt=ti.f32, shape=())

fov = ti.var(dt=ti.f32, shape=())
aspect = ny / nx
cam_r = ti.var(dt=ti.f32, shape=())
cam_theta = ti.var(dt=ti.f32, shape=())
cam_phi = ti.var(dt=ti.f32, shape=())

cam_lower_left_corner = ti.Vector(3, dt=ti.f32, shape=())
cam_horizontal = ti.Vector(3, dt=ti.f32, shape=())
cam_vertical = ti.Vector(3, dt=ti.f32, shape=())
cam_origin = ti.Vector(3, dt=ti.f32, shape=())

dir_l_pos = ti.Vector(3, dt=ti.f32, shape=())
dir_l_theta = ti.var(dt=ti.f32, shape=())
dir_l_phi = ti.var(dt=ti.f32, shape=())

sphere_num = 1000
sphere_origin_list = ti.Vector(3, dt=ti.f32, shape=(sphere_num))
sphere_radius_list = ti.var(dt=ti.f32, shape=(sphere_num))
sphere_material_color_list = ti.Vector(3, dt=ti.f32, shape=(sphere_num))

@ti.kernel
def init():
    up[None] = [0.0, 1.0, 0.0]
    lookat[None] = [0.0, 0.0, 0.0]
    fov[None] = math.pi / 3
    cam_r[None] = 2.0
    cam_theta[None] = math.pi / 4
    cam_phi[None] = math.pi / 4
    dir_l_theta[None] = math.pi / 4
    dir_l_phi[None] = math.pi / 4

    for i in range(sphere_num):
        sphere_origin_list[i] = ti.Vector([ti.random() * 2 - 1, ti.random() * 2 - 1, ti.random() * 2 - 1])
        sphere_radius_list[i] = 0.03
        sphere_material_color_list[i] = ti.Vector([1, 1, 1])

@ti.func
def normalize(v):
    return v.normalized()

@ti.kernel
def generate_parameters():
    lookfrom[None] = [cam_r[None] * ti.sin(cam_theta[None]) * ti.sin(cam_phi[None]),
                      cam_r[None] * ti.cos(cam_phi[None]),
                      cam_r[None] * ti.cos(cam_theta[None]) * ti.sin(cam_phi[None])] + lookat[None]
    dir_l_pos[None] = [ti.sin(dir_l_theta[None]) * ti.sin(dir_l_phi[None]),
                       ti.cos(dir_l_phi[None]),
                       ti.cos(dir_l_theta[None]) * ti.sin(dir_l_phi[None])]
    w = normalize(lookfrom[None] - lookat[None])
    u = normalize(up[None].cross(w))
    v = w.cross(u)

    half_height = ti.tan(fov / 2.0)
    half_width = aspect * half_height
    cam_origin[None] = lookfrom[None]
    cam_lower_left_corner[None] = cam_origin - half_width * u - half_height * v - w
    cam_horizontal[None] = 2 * half_width * u
    cam_vertical[None] = 2 * half_height * v

@ti.func
def cam_get_ray(u, v):
    return cam_origin, cam_lower_left_corner[None] + u * cam_horizontal[None] + v * cam_vertical[None] - cam_origin

@ti.func
def hit_sphere(sphere_center, sphere_radius, ray_origin, ray_direction, t_min,
               t_max):
    oc = ray_origin - sphere_center
    a = ray_direction.dot(ray_direction)
    b = oc.dot(ray_direction)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    discriminant = b * b - a * c

    hit_flag = False
    hit_t = 0.0
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    if discriminant > 0.0:
        temp = (-b - ti.sqrt(b * b - a * c)) / a
        if temp < t_max and temp > t_min:
            hit_t = temp
            hit_p = ray_origin + hit_t * ray_direction
            hit_normal = (hit_p - sphere_center) / sphere_radius
            hit_flag = True
        if hit_flag == False:
            temp = (-b + ti.sqrt(b * b - a * c)) / a
            if temp < t_max and temp > t_min:
                hit_t = temp
                hit_p = ray_origin + hit_t * ray_direction
                hit_normal = (hit_p - sphere_center) / sphere_radius
                hit_flag = True
    return hit_flag, hit_t, hit_p, hit_normal

@ti.func
def hit_all_spheres(ray_origin, ray_direction, t_min, t_max):
    hit_anything = False
    hit_t = 0.0
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_material_color = ti.Vector([0.0, 0.0, 0.0])
    closest_so_far = t_max
    for i in range(sphere_num):
        hit_flag, temp_hit_t, temp_hit_p, temp_hit_normal = \
            hit_sphere(sphere_origin_list[i], sphere_radius_list[i], ray_origin, ray_direction, t_min, closest_so_far)
        if hit_flag:
            hit_anything = True
            closest_so_far = temp_hit_t
            hit_t = temp_hit_t
            hit_p = temp_hit_p
            hit_normal = temp_hit_normal
            hit_material_color = sphere_material_color_list[i]
    return hit_anything, hit_t, hit_p, hit_normal, hit_material_color

@ti.func
def hit_any_spheres(ray_origin, ray_direction, t_min, t_max):
    result = False
    for i in range(sphere_num):
        hit_flag, temp_hit_t, temp_hit_p, temp_hit_normal = \
            hit_sphere(sphere_origin_list[i], sphere_radius_list[i], ray_origin, ray_direction, t_min, t_max)
        if hit_flag:
            result = True
            break
    return result

@ti.func
def color(ray_origin, ray_direction):
    col = ti.Vector([0.0, 0.0, 0.0])
    hit_flag, hit_t, hit_p, hit_normal, hit_material_color = \
        hit_all_spheres(ray_origin, ray_direction, 0.001, 10e9)
    if hit_flag:
        inShadow = hit_any_spheres(hit_p, dir_l_pos, 0.001, 10e9)
        dot = hit_normal.dot(dir_l_pos)
        if inShadow or dot <= 0.05:
            col = 0.05 * hit_material_color
        else:
            col = dot * hit_material_color
    else:
        unit_direction = normalize(ray_direction)
        t = 0.5 * (unit_direction.y + 1.0)
        col = (1.0 - t) * ti.Vector([0.7, 0.9, 1.0]) + t * ti.Vector([0.3, 0.6, 0.8])
    return col


@ti.kernel
def render():
    for i, j in pixels:
        u = i / nx #(i + ti.random()) / nx
        v = j / ny #(j + ti.random()) / ny
        ray_origin, ray_direction = cam_get_ray(u, v)
        pixels[i, j] = color(ray_origin, ray_direction)

def main():
    init()
    gui = ti.GUI("sphere trace", (nx, ny))
    while gui.running:
        gui.get_event()
        if gui.is_pressed('s') and cam_phi[None] > math.pi / 32:
            cam_phi[None] -= math.pi / 32
        elif gui.is_pressed('w') and cam_phi[None] < math.pi - math.pi / 32:
            cam_phi[None] += math.pi / 32
        elif gui.is_pressed('a'):
            cam_theta[None] += math.pi / 32
            while cam_theta[None] > 2 * math.pi:
                cam_theta[None] -= 2 * math.pi
        elif gui.is_pressed('d'):
            cam_theta[None] -= math.pi / 32
            while cam_theta[None] < 0:
                cam_theta[None] += 2 * math.pi
        elif gui.is_pressed(ti.GUI.UP) and cam_r[None] > 0.1:
            cam_r[None] /= 1.1
        elif gui.is_pressed(ti.GUI.DOWN):
            cam_r[None] *= 1.1
        elif gui.is_pressed(ti.GUI.LEFT) and fov[None] > math.pi / 96:
            fov[None] -= math.pi / 96
        elif gui.is_pressed(ti.GUI.RIGHT) and fov[None] < math.pi - math.pi / 96:
            fov[None] += math.pi / 96
        elif gui.is_pressed('i') and dir_l_phi[None] > -math.pi / 2 + math.pi / 32:
            dir_l_phi[None] -= math.pi / 32
        elif gui.is_pressed('k') and dir_l_phi[None] < math.pi / 2 - math.pi / 32:
            dir_l_phi[None] += math.pi / 32
        elif gui.is_pressed('j'):
            dir_l_theta[None] += math.pi / 32
            while dir_l_theta[None] > 2 * math.pi:
                dir_l_theta[None] -= 2 * math.pi
        elif gui.is_pressed('l'):
            dir_l_theta[None] -= math.pi / 32
            while dir_l_theta[None] < 0:
                dir_l_theta[None] += 2 * math.pi
        elif gui.is_pressed(ti.GUI.ESCAPE):
            gui.running = False

        generate_parameters()
        render()
        gui.set_image(np.sqrt(pixels.to_numpy()))
        gui.show()

if __name__ == '__main__':
    main()