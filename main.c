#include "k_means_functions.h"

#define N 2000000               // Number of points in the database
#define N_CENTROIDS 10          // Number of centroids you want to use
#define MAX_THR 8               // NUmber of threads you want to activate

#define DIM_ARRAY 2             // Dimension of the array of the points you want to use
// to profile the parallel version. (You need gnuplot!)


int main()
{
    //#################################################################################################################################
    // For a single execution you can run this:

    srand(time(NULL));
    double start, stop;
    Point *true_centroids = centroids_initialize(N_CENTROIDS);                      // True centroids used to create the database
    Point* database = new_2D_database(N,N_CENTROIDS, true_centroids);
    Point* centroids = centroids_initialize(N_CENTROIDS);                           // First random initialization of the centroids


    // Sequential verison, that uses only one active thread
    start = omp_get_wtime();
    find_2D_centroids(database, centroids, N, N_CENTROIDS, 1);
    stop = omp_get_wtime();
    printf("SEQUENTIAL TIME: %lf s\n", stop - start);


    // Parallel version, that use the number of threads you have selected
    centroids = centroids_initialize(N_CENTROIDS);
    start = omp_get_wtime();
    find_2D_centroids(database, centroids, N, N_CENTROIDS, MAX_THR);
    stop = omp_get_wtime();
    printf("PARALLEL TIME:   %lf s\n", stop - start);


    //graph(database, centroids, true_centroids, N, N_CENTROIDS);                      //This line requires gnuplot!


    free(true_centroids);
    free(database);
    free(centroids);


    //#################################################################################################################################
    // If you want to profile the performance you need to run these lines. YOU NEED GNUPLOT!
    /*
    int number_of_points[DIM_ARRAY] = {100000,2000000};                         // These are the number of points (of the database)
                                                                                // you want to generate for the different executions.
                                                                                // You can add any number you want, in ascending order!

    speedup_curves(number_of_points, N_CENTROIDS, MAX_THR, DIM_ARRAY);          // It plots the graphs
    */

    //#################################################################################################################################
    //# NOTE: in order to have comparable times, the function FIND_2D_CENTROID executes 40 cycles before stopping,                    #
    //#       for each call.                                                                                                          #
    //#                                                                                                                               #
    //# For more information, read the functions' description!                                                                        #
    //#################################################################################################################################
    return 0;
}