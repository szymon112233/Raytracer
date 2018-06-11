#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <ctime>

#if defined __linux__ || defined __APPLE__
// "Compiled for Linux
#else
// Windows doesn't define these values by default, Linux does
#define M_PI 3.141592653589793
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

template<typename T>
class Vec3
{
public:
    T x, y, z;
    CUDA_CALLABLE_MEMBER Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    CUDA_CALLABLE_MEMBER Vec3(T xx) : x(xx), y(xx), z(xx) {}
    CUDA_CALLABLE_MEMBER Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    CUDA_CALLABLE_MEMBER Vec3& normalize()
    {
        T nor2 = length2();
        if (nor2 > 0) {
            T invNor = 1 / sqrt(nor2);
            x *= invNor, y *= invNor, z *= invNor;
        }
        return *this;
    }
    CUDA_CALLABLE_MEMBER Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
    CUDA_CALLABLE_MEMBER Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
    CUDA_CALLABLE_MEMBER T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
    CUDA_CALLABLE_MEMBER Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
    CUDA_CALLABLE_MEMBER Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
    CUDA_CALLABLE_MEMBER Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
    CUDA_CALLABLE_MEMBER Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
    CUDA_CALLABLE_MEMBER Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
    CUDA_CALLABLE_MEMBER T length2() const { return x * x + y * y + z * z; }
    CUDA_CALLABLE_MEMBER T length() const { return sqrt(length2()); }
    friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
    {
        os << "[" << v.x << " " << v.y << " " << v.z << "]";
        return os;
    }
};

typedef Vec3<float> Vec3f;

class Sphere
{
public:
    Vec3f center;                           /// position of the sphere
    float radius, radius2;                  /// sphere radius and radius^2
    Vec3f surfaceColor, emissionColor;      /// surface color and emission (light)
    float transparency, reflection;         /// surface transparency and reflectivity
    Sphere(){}
	Sphere(
        const Vec3f &c,
        const float &r,
        const Vec3f &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vec3f &ec = 0) :
        center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
        transparency(transp), reflection(refl)
    { /* empty */ }
    // Compute a ray-sphere intersection using the geometric solution
    CUDA_CALLABLE_MEMBER bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const
    {
        Vec3f l = center - rayorig;
        float tca = l.dot(raydir);
        if (tca < 0) return false;
        float d2 = l.dot(l) - tca * tca;
        if (d2 > radius2) return false;
        float thc = sqrt(radius2 - d2);
        t0 = tca - thc;
        t1 = tca + thc;
        
        return true;
    }
};

// This variable controls the maximum recursion depth
#define MAX_RAY_DEPTH 5

CUDA_CALLABLE_MEMBER float mix(const float &a, const float &b, const float &mix)
{
    return b * mix + a * (1 - mix);
}

// This is the main trace function. It takes a ray as argument (defined by its origin
// and direction). We test if this ray intersects any of the geometry in the scene.
// If the ray intersects an object, we compute the intersection point, the normal
// at the intersection point, and shade this point using this information.
// Shading depends on the surface property (is it transparent, reflective, diffuse).
// The function returns a color for the ray. If the ray intersects an object that
// is the color of the object at the intersection point, otherwise it returns
// the background color.
CUDA_CALLABLE_MEMBER
Vec3f trace(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    Sphere *spheres,
	int spheresSize,
    const int &depth)
{
    //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
    float tnear = INFINITY;
    const Sphere* sphere = NULL;
    // find intersection of this ray with the sphere in the scene
    for (int i = 0; i < spheresSize; ++i) {
        float t0 = INFINITY, t1 = INFINITY;
        if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
            if (t0 < 0) t0 = t1;
            if (t0 < tnear) {
                tnear = t0;
                sphere = &spheres[i];
            }
        }
    }
    // if there's no intersection return black or background color
    if (!sphere) return Vec3f(2);
    Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
    Vec3f phit = rayorig + raydir * tnear; // point of intersection
    Vec3f nhit = phit - sphere->center; // normal at the intersection point
    nhit.normalize(); // normalize normal direction
    // If the normal and the view direction are not opposite to each other
    // reverse the normal direction. That also means we are inside the sphere so set
    // the inside bool to true. Finally reverse the sign of IdotN which we want
    // positive.
    float bias = 1e-4; // add some bias to the point from which we will be tracing
    bool inside = false;
    if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
    /*if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
        float facingratio = -raydir.dot(nhit);
        // change the mix value to tweak the effect
        float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
        // compute reflection direction (not need to normalize because all vectors
        // are already normalized)
        Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
        refldir.normalize();
        Vec3f reflection = trace(phit + nhit * bias, refldir, spheres, spheresSize, depth + 1);
        Vec3f refraction = 0;
        // if the sphere is also transparent compute refraction ray (transmission)
        if (sphere->transparency) {
            float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
            float cosi = -nhit.dot(raydir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vec3f refrdir = raydir * eta + nhit * (eta *  cosi - sqrt(k));
            refrdir.normalize();
            refraction = trace(phit - nhit * bias, refrdir, spheres, spheresSize, depth + 1);
        }
        // the result is a mix of reflection and refraction (if the sphere is transparent)
        surfaceColor = (
            reflection * fresneleffect +
            refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
    }
    else {*/
        // it's a diffuse object, no need to raytrace any further
        for (int i = 0; i < spheresSize; ++i) {
            if (spheres[i].emissionColor.x > 0) {
                // this is a light
                Vec3f transmission = 1;
                Vec3f lightDirection = spheres[i].center - phit;
                lightDirection.normalize();
                for (int j = 0; j < spheresSize; ++j) {
                    if (i != j) {
                        float t0, t1;
                        if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
                            transmission = 0;
                            break;
                        }
                    }
                }
                surfaceColor += sphere->surfaceColor * transmission *
                max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
            }
        }
    //}
    
    return surfaceColor + sphere->emissionColor;
}

__global__ void fill_image(Sphere *spheres, int spheresSize, Vec3f *image, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
	float fov = 30;
    float angle = tan(M_PI * 0.5 * fov / 180.);
    float aspectratio = width / float(height);
	for (int i = index; i < width * height; i += stride) {
		int y = i / width;
		int x = i - y * width;
        float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
        float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
        Vec3f raydir(xx, yy, -1);
        raydir.normalize();
        image[i] = trace(Vec3f(0), raydir, spheres, spheresSize, 0);
	}
}

// Main rendering function. We compute a camera ray for each pixel of the image
// trace it and return a color. If the ray hits a sphere, we return the color of the
// sphere at the intersection point, else we return the background color.
void render(Sphere *spheres, int spheresSize)
{
    unsigned width = 640, height = 480;
	int blockSize = 256;
	int numBlocks = (width * height + blockSize - 1) / blockSize;
    Vec3f *image;
	cudaMallocManaged(&image, height * width * sizeof(Vec3f));
    // Trace rays
	clock_t begin = clock();
	fill_image<<<numBlocks, blockSize>>>(spheres, spheresSize, image, width, height);
	cudaDeviceSynchronize();
	clock_t end = clock();
	printf("Elapsed time: %lf\n", double(end - begin) / CLOCKS_PER_SEC);
    // Save result to a PPM image (keep these flags if you compile under Windows)
    std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < width * height; ++i) {
        ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
               (unsigned char)(std::min(float(1), image[i].y) * 255) <<
               (unsigned char)(std::min(float(1), image[i].z) * 255);
    }
    ofs.close();
	cudaFree(image);
}

// In the main function, we will create the scene which is composed of 5 spheres
// and 1 light (which is also a sphere). Then, once the scene description is complete
// we render that scene, by calling the render() function.
int main(int argc, char **argv)
{
    srand(13);
	int spheresSize = 6;
	Sphere *spheres;
	cudaMallocManaged(&spheres, spheresSize * sizeof(Sphere));
    // position, radius, surface color, reflectivity, transparency, emission color
    spheres[0] = Sphere(Vec3f( 0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0);
    spheres[1] = Sphere(Vec3f( 0.0,      0, -20),     4, Vec3f(1.00, 0.32, 0.36), 1, 0.5);
    spheres[2] = Sphere(Vec3f( 5.0,     -1, -15),     2, Vec3f(0.90, 0.76, 0.46), 1, 0.0);
    spheres[3] = Sphere(Vec3f( 5.0,      0, -25),     3, Vec3f(0.65, 0.77, 0.97), 1, 0.0);
    spheres[4] = Sphere(Vec3f(-5.5,      0, -15),     3, Vec3f(0.90, 0.90, 0.90), 1, 0.0);
    // light
    spheres[5] = Sphere(Vec3f( 0.0,     20, -30),     3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3));
    render(spheres, spheresSize);
	cudaFree(spheres);
    
    return 0;
}
