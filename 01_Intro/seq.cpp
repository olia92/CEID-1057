#include <cstdio>
#include <iostream>
#include <vector>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif


#include <omp.h>



int main(int argc, char** argv )
{
    // vector size
    const int N = 1600000;

    double start_time, end_time;
    
    
    // initialize vectors
    std::vector<float> x(N,-1.2), y(N,3.4), z(N);
start_time = omp_get_wtime(); // Start the timer
    
    // do the sum z = x + y
    for(int i = 0; i < N; i++) z[i] = x[i] + y[i];

   end_time = omp_get_wtime(); // End the timer
    
    std::cout << "Time taken: " << end_time - start_time << " seconds" << std::endl;
    
    start_time = omp_get_wtime();
    // DO THE SUM z = x + y with SSE (width=4)
        for( int i = 0; i < N; i += 4 ){
            // z[i] = x[i] + y[i];
            __m128 xx = _mm_load_ps( &x[i] );
            __m128 yy = _mm_load_ps( &y[i] );
            __m128 zz = _mm_add_ps( xx, yy );
            _mm_store_ps( &z[i], zz );
            }
    end_time = omp_get_wtime();

    std::cout << "With SSE: " << end_time - start_time << " seconds" << std::endl;
    

    // for(int i = 0; i < N; i++){
    //     std::cout<<i<<". "<<z[i]<<" = "<<x[i]<<" + "<<y[i]<<std::endl;
    // }
    return 0;
}

/*
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t start = 1000000L * tv.tv_sec + tv.tv_usec;

    run_classification(samples, num_samples, NULL);

    gettimeofday(&tv,NULL);
    uint64_t end = 1000000L * tv.tv_sec + tv.tv_usec;
    printf("%ld microseconds\n", end - start);

*/

/*
    double start_time, end_time;
    
    start_time = omp_get_wtime(); // Start the timer
    
    // Execute the process
    #pragma omp parallel for
    for(int i = 0; i < 10000000; i++) {
        // some computation here
    }
    
    end_time = omp_get_wtime(); // End the timer
    
    std::cout << "Time taken: " << end_time - start_time << " seconds" << std::endl;
    
*/