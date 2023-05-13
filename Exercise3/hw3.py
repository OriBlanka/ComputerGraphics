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

            color = calc_I(camera, objects, lights, ambient, ray, max_depth)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image

# ray is the ray from the point of view to the directiojn of the pixel/ the reflected ray
def calc_I(origin, objects, lights, ambient, ray, max_depth, current_depth=0):
    if current_depth == max_depth:
        return np.zeros(3)

    nearest_object, distance_to_nearest_object = ray.nearest_intersected_object(objects=objects)
    if nearest_object:

        K_A = nearest_object.ambient
        I_A = ambient
        K_D = np.array(nearest_object.diffuse)

        object_intersection_point = origin + distance_to_nearest_object * ray.direction

        N = normalize(object_intersection_point - nearest_object.center) if isinstance(nearest_object, helper_classes.Sphere) else \
            normalize(nearest_object.normal)
        K_S = np.array(nearest_object.specular)
        n = nearest_object.shininess

        object_intersection_point += EPSILON * N

        Sigma_L = 0

        for light in lights:
            object_intersection_to_light_vector = light.get_light_ray(object_intersection_point).direction
            object_intersection_to_light_ray = Ray(origin=object_intersection_point, direction=object_intersection_to_light_vector)

            L = object_intersection_to_light_vector
            V = normalize(origin - object_intersection_point)  # normalize(origin - object_intersection_point)
            R = normalize(reflected(vector=-L, axis=N))

            S_L = get_S_L(objects, object_intersection_point, object_intersection_to_light_ray, light)
            I_L = light.get_intensity(object_intersection_point)
            Sigma_L += ((K_D * dot(N, L)) + (K_S * (dot(V, R) ** n))) * S_L * I_L

        K_R = nearest_object.reflection
        reflected_ray = calculate_reflected_ray(vector=ray.direction, intersection_point=object_intersection_point, normal_vector=N)
        I_R = calc_I(origin=object_intersection_point, objects=objects, lights=lights, ambient=ambient, ray=reflected_ray,
                     max_depth=max_depth, current_depth=current_depth + 1)
        I = K_A * I_A + Sigma_L + K_R * I_R

    else:
        I = np.zeros(3)

    return I

def get_S_L(objects, object_intersection_point, object_intersection_to_light_ray, light):
    closest_object, minimum_distance_to_object = object_intersection_to_light_ray.nearest_intersected_object(objects)

    return 0 if minimum_distance_to_object < light.get_distance_from_light(object_intersection_point) else 1


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    return camera, lights, objects
