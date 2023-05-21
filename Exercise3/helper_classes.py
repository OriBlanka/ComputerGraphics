import math

import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    v = np.array([0, 0, 0])
    v = normalize(vector) - 2 * (np.dot(normalize(vector), axis)) * axis
    return v

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        pass

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = direction
        self.kc = kc
        self.kl = kl
        self.kq = kq


    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        vec = self.position - intersection
        f_att = self.kc + self.kl * d + self.kq * (d ** 2)
        return self.intensity * np.dot(normalize(vec), normalize(self.direction)) / f_att


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf
        distance_object_tuple = None
        intersections = {}

        for obj in objects:
            if obj.intersect(self):
                distance_object_tuple = obj.intersect(self)
                intersections[distance_object_tuple[1]] = distance_object_tuple[0]

            if distance_object_tuple and distance_object_tuple[0] < min_distance:
                min_distance, nearest_object = distance_object_tuple

        if len(intersections) == 0:
            return None


        return min_distance, nearest_object


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None


class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |
         |               |
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.normal = self.compute_normal()

    def compute_normal(self):
        ab = self.abcd[1] - self.abcd[0]
        ad = self.abcd[3] - self.abcd[0]
        n = np.cross(ab, ad)
        return normalize(n)

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        n = self.compute_normal()
        plane = Plane(n, self.abcd[0])

        if plane.intersect(ray):
            t, _ = plane.intersect(ray)
            p = ray.origin + t * ray.direction

            for i in range(4):
                p1 = self.abcd[i] - p
                p2 = self.abcd[(i + 1) % 4] - p
                dot_product = np.dot(n, np.cross(p1, p2))
                if not dot_product > 0:
                    return None

            return t, self
        return None


class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """
        A = B = C = D = E = F = None
        self.face_list = [A, B, C, D, E, F]
        self.vertices = [a, b, c, d, e, f]
        self.create_rectangles_faces()

    def create_rectangles_faces(self):
        a, b, c, d, e, f = self.vertices
        h = [e[0] - (c[0] - b[0]), e[1] - (c[1] - b[1]), e[2] - (c[2] - b[2])]
        g = [f[0] - (d[0] - a[0]), f[1] - (d[1] - a[1]), f[2] - (d[2] - a[2])]
        A = Rectangle(a, b, c, d)
        B = Rectangle(d, c, e, f)
        C = Rectangle(g, h, e, f)
        D = Rectangle(a, b, h, g)
        E = Rectangle(g, a, d, f)
        F = Rectangle(h, b, c, e)

        self.face_list = [A, B, C, D, E, F]

    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        return ray.nearest_intersected_object(self.face_list)

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        # Calculate vector from ray origin to sphere center
        ray_to_center = self.center - ray.origin

        # Calculate dot product of vector and ray direction
        vector_dot_product_ray_direction = np.dot(ray_to_center, ray.direction)

        # Calculate discriminant
        discriminant = vector_dot_product_ray_direction ** 2 - np.sum(np.power(ray_to_center, 2)) + self.radius ** 2

        if discriminant < 0:
            return None

        t1result = vector_dot_product_ray_direction - math.sqrt(discriminant)
        t2result = vector_dot_product_ray_direction + math.sqrt(discriminant)

        if t1result < 0 and t2result < 0:
            return None
        else:
            return min(t1result, t2result), self
