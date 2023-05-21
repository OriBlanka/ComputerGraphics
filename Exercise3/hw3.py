from numpy import dot

import helper_classes
from helper_classes import *
import matplotlib.pyplot as plt


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            if ray.nearest_intersected_object(objects):
                min_distance, nearest_object = ray.nearest_intersected_object(objects)
                p = ray.origin + min_distance * ray.direction

                # getColor
                color = get_color(ray, ambient, nearest_object, p, lights, objects, max_depth, 1)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image

# TODO: Change function to our version
def get_color(ray, ambient, obj, hitP, lights, objects, max_depth, level):
    sigma = 0
    normal = None
    if isinstance(obj, Sphere):
        normal = normalize(hitP - obj.center)
    else:
        normal = obj.normal

    hitP = hitP + 0.01 * normal
    for light in lights:
        light_ray = light.get_light_ray(hitP)
        I_L = light.get_intensity(hitP)
        K_D = obj.diffuse
        K_S = obj.specular
        N = normal
        L = light.get_light_ray(hitP).direction
        V = normalize(ray.origin - hitP)
        L_R = reflected(L, N)
        s_j = 1
        if light_ray.nearest_intersected_object(objects):
            min_distance, nearest_object = light_ray.nearest_intersected_object(objects)
            s_j = calculate_s_j(ray, hitP, min_distance, nearest_object)
        sigma = sigma + s_j * ((K_D * I_L * np.dot(N, L)) + (K_S * I_L * (np.dot(V, L_R) ** (obj.shininess / 10))))

    I_amb = ambient
    K_A = obj.ambient
    color = K_A * I_amb + sigma

    level = level + 1
    if level > max_depth:
        return color

    # reflection
    # construct new_ray
    new_direction = reflected(ray.direction, normal)
    reflected_ray = Ray(hitP, new_direction)
    # construct new hitP
    if reflected_ray.nearest_intersected_object(objects):
        min_distance, nearest_object = reflected_ray.nearest_intersected_object(objects)
        new_hitP = reflected_ray.origin + min_distance * reflected_ray.direction
        if isinstance(nearest_object, Sphere):
            normal = normalize(new_hitP - nearest_object.center)
        else:
            normal = nearest_object.normal
        new_hitP = new_hitP + 0.01 * normal
        # recursion call
        color = color + obj.reflection * get_color(reflected_ray, ambient, nearest_object, new_hitP, lights, objects, max_depth, level)
    return color

# TODO: Change function to our version
def calculate_s_j(ray, hitP, min_distance, nearest_object):
    s_j = 1
    distance_from_camera = np.linalg.norm(ray.origin - hitP)
    if nearest_object and distance_from_camera > min_distance:
        s_j = 0

    return s_j

# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    return camera, lights, objects
