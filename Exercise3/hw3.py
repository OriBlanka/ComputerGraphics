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
        vector_a = hitP - obj.center
        normal = normalize(vector_a)
    else:
        normal = obj.normal


    calacNormalScal = 0.01 * normal
    hitP = hitP + calacNormalScal

    for light in lights:
        light_Ray = light.get_light_ray(hitP)
        Intensity_L = light.get_intensity(hitP)
        K_Diffuse = obj.diffuse
        K_Specular = obj.specular
        Norm = normal
        Ligh = light.get_light_ray(hitP).direction
        VectorPreNormilized = ray.origin - hitP
        VectorNormalized = normalize(VectorPreNormilized)
        L_Reflected = reflected(Ligh, Norm)
        s_j = 1
        if light_Ray.nearest_intersected_object(objects):
            min_distance, nearest_object = light_Ray.nearest_intersected_object(objects)
            s_j = calculate_s_j(ray, hitP, min_distance, nearest_object)
        calc1 = (K_Diffuse * Intensity_L * np.dot(Norm, Ligh))
        calc2 = (K_Specular * Intensity_L * (np.dot(VectorNormalized, L_Reflected)))
        calc3 = (obj.shininess / 10)
        sigma = sigma + s_j * (calc1 + calc2 ** calc3)


    I_ambient = ambient
    K_ambient = obj.ambient
    ambientMulti = K_ambient * I_ambient
    color = ambientMulti+ sigma

    level = level + 1
    if level > max_depth:
        return color

    # reflection
    # construct new_ray
    reflected_ray = Ray(hitP, reflected(ray.direction, normal))
    # construct new hitP
    if reflected_ray.nearest_intersected_object(objects):
        min_distance, nearest_object = reflected_ray.nearest_intersected_object(objects)
        Calc4 = min_distance * reflected_ray.direction
        new_HitPoint = reflected_ray.origin + Calc4

        if isinstance(nearest_object, Sphere):
            vectorcalcpreNorm = new_HitPoint - nearest_object.center
            normal = normalize(vectorcalcpreNorm)
        else:
            normal = nearest_object.normal
        scalernorm = 0.01 * normal
        new_hitpoint = new_HitPoint + scalernorm
        # recursion call

        oclor = get_color(reflected_ray, ambient, nearest_object, new_hitpoint, lights, objects, max_depth, level)
        color = color + obj.reflection * oclor
    return color

# TODO: Change function to our version
def calculate_s_j(ray, hitP, min_distance, nearest_object):
    s_j = 1
    vectorprenorm = ray.origin - hitP
    distance_from_camera = np.linalg.norm(vectorprenorm)

    distancecheck = distance_from_camera > min_distance
    if nearest_object and distancecheck:
        s_j = 0

    return s_j

# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0, 0, 1])

    plane = Plane([0, 1, 0], [0, -0.3, 0])
    plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)

    sphere_a = Sphere([-0.5, 0.2, -1], 0.5)
    sphere_a.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)

    sphere_b = Sphere([0.6, 0.5, -0.5], 0.8)
    sphere_b.set_material([0, 1, 0], [0, 1, 0], [0.3, 0.3, 0.3], 100, 0.2)

    cuboid = Cuboid(
        [-1, -.75, -2],
        [-1, -2, -2],
        [1, -2, -1.5],
        [1, -.75, -1.5],
        [2, -2, -2.5],
        [2, -.75, -2.5]
    )

    cuboid.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
    cuboid.apply_materials_to_faces()

    light1 = PointLight(intensity=np.array([1, 1, 1]), position=np.array([1, 1.5, 1]), kc=0.1, kl=0.1, kq=0.1)
    light2 = DirectionalLight(intensity= np.array([1, 1, 1]),direction=np.array([1,1,1]))
    lights = [light1, light2]
    objects = [sphere_a, sphere_b, cuboid, plane]
    return camera, lights, objects
