#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    // ox, oy：光线在观察平面（Z=0平面）上的起点坐标。
    // *n：一个指针，用于返回击中点处表面法向量的 Z分量
    __device__ float hit( float ox, float oy, float *n ) {
        // 光线在xy平面内与球心距离
        float dx = ox - x;
        float dy = oy - y;
        // 如果命中
        if (dx*dx + dy*dy < radius*radius) {
            // dz是光线与球交点到xy平面的距离
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            // 标准化
            *n = dz / sqrtf( radius * radius );
            // 如果使用下表面进行计算, 法向量n是负数
            // *n = - dz / sqrtf( radius * radius );
            // TODO 为什么这里是z+dz而不是z-dz
            // 是一个简化的模型, 从原点向上看,每个球只渲染上半球
            // 因为对于上半球，交点处的法向量Z分量 n (= dz / radius) 是正数，这才能够用作有效的亮度缩放因子 fscale。
            // 它假设你的视线“穿透”了球体的下半部分，直接“看到”了被照亮的上半部分曲面。
            // 在这种设定下，光线可以被理解为：从相机发出，穿过一个“透明”的下半球，然后击中并被上半球表面反射回来。
            return dz + z;
            // 如果观察球的下表面 ,可能有一部分球穿过xoy平面
            // return z-dz;
        }
        return -INF;
        // return INF;
    }
};
#define SPHERES 20


__global__ void kernel( Sphere *s, unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;
    float   minz = INF;
    // 遍历每个球 ,t为z+dz 即光线与球面交点到xoy平面距离
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = s[i].hit( ox, oy, &n );
        if (t > maxz) {
            // fscale 是 颜色缩放系数
            // 光线与球表面交点的法向量 n -> fscale
            // 光照模型 : 朗伯漫反射
            // 视觉效果：因此，fscale = n 直接给出了光照强度。
                // 当光线垂直照射在球体正顶部时，法向量是 (0, 0, 1)，n = 1，颜色完全保留，最亮。
                // 当光线照射在侧面时，法向量倾斜，n 值在 0 到 1 之间，颜色按比例变暗。
                // 当光线照射在垂直于光线的边缘时，n ≈ 0，颜色几乎变为黑色。
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
        // 这个是真实的逻辑,渲染离观察点最近的, 但是还要和其他几处一并修改,才有作用
        // if (t < minz){
        //     // float fscale = n;
        //     float fscale = fabs(n);
        //     r = s[i].r * fscale;
        //     g = s[i].g * fscale;
        //     b = s[i].b * fscale;
        //     minz = t;
        // }
    } 
    // rgba存储
    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}


// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
    Sphere          *s;
};

int main( void ) {
    DataBlock   data;
    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char   *dev_bitmap;
    Sphere          *s;


    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() ) );
    // allocate memory for the Sphere dataset
    HANDLE_ERROR( cudaMalloc( (void**)&s,
                              sizeof(Sphere) * SPHERES ) );

    // allocate temp memory, initialize it, copy to
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }
    HANDLE_ERROR( cudaMemcpy( s, temp_s,
                                sizeof(Sphere) * SPHERES,
                                cudaMemcpyHostToDevice ) );
    free( temp_s );

    // generate a bitmap from our sphere data
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>( s, dev_bitmap );

    // copy our bitmap back from the GPU for display
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    // 1.6ms
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    HANDLE_ERROR( cudaFree( s ) );

    // display
    bitmap.display_and_exit();
}

