#include "k_means_functions.h"



#define MAX 10000           // max value of your points set
#define MIN 0               // min value of your points set
#define DEV 1000            // deviation I will use to create the database starting from centroids

#define MAX_DIST sqrt(2)    // I will normalize the dataset, so the max distance between 2 different point
                            // will be 2^(1/2)

#define PRECISION 0.000001  // It will be used to evaluate the precision of the founded centroids




Point* centroids_initialize(int k){
    /*
     * k = number of centroids you want to initialize
     *
     * DESCRIPTION:
     * It initializes a struct Point array of k dimension with normalized random point (in [0;1] interval)
     */

    Point *centroids = (Point*) malloc(k*sizeof (Point));

    for(int i = 0; i < k; i++){
        centroids[i].x = (double)((rand()%(MAX-MIN+1))+MIN)/(MAX-MIN);
        centroids[i].y = (double)((rand()%(MAX-MIN+1))+MIN)/(MAX-MIN);
    }

    return centroids;
}

Point* new_2D_database( int n, int k, Point *centroids) {
    /*
     * n = number of point in the database
     * k = number of centroids you want to use
     * centroids = struct Point array with centroids you want to use
     *
     * DESCRIPTION:
     * It takes the true centroids you want to use for your database and for each of them
     * it produces n/k point with deviation DEV (x_1 +/- DEV; y_1 +/- DEV).
     * To use the function you need OpenMP
     */

    Point *points = (Point*) malloc(n*sizeof (Point));
    if (points == NULL) return NULL;

    // if you have more centroids than point return NULL
    if (k > n){
        printf("ATTENTION! YOU HAVE MORE CENTROIDS THAN POINTS");
        return NULL;
    }

#pragma omp parallel default(none) shared(points, n, k) firstprivate(centroids)
    {

        int id = omp_get_thread_num();
        int tot_thr = omp_get_num_threads();
        int dim_c = (int) (n / k), dim_m = (int) (n / tot_thr);     // Divide points array in 'tot_thr' parts
        int start = id * dim_m, stop = (id + 1)*dim_m;              // Preparing start and stop for the next loop
        if(id == tot_thr - 1) stop = n;                             // For the last thread stop == n

        int n_chunk = (int) (start / dim_c);                        // First centroid every thread will use


        for (int i = start; i < stop; ++i ){
            points[i].x = centroids[n_chunk].x + (double)((rand()%(2*DEV+1))-DEV)/(MAX-MIN);
            points[i].y = centroids[n_chunk].y + (double)((rand()%(2*DEV+1))-DEV)/(MAX-MIN);

            // When n/k points are created for a single centroid, function passes to the next centroid
            if ((i+1)%dim_c == 0) {
                if (n_chunk + 1 < k) n_chunk++;
            }
        }
    }
    return points;
}

void find_2D_centroids(Point* db, Point* centroids, int n, int k, int num_thr) {
    /*
     * db = struct Point array of dimension n. It is the database the function will use
     * centroids = struct Point array of dimension k. These are the starting random centroids
     * n = number of points in database
     * k = number of centroids
     * num_thr = number of threads you want to activate
     *
     * DESCRIPTION:
     * It takes the database db and the starting random centroids, and it returns the calculated centroids.
     * To use the function you need OpenMP.
     */


#pragma omp parallel num_threads(num_thr) default(none) shared(n, num_thr, centroids, db, k)
    {
#pragma omp single
        {

            int counter, iterations = 0, my_exit = 0;       // Control variables.
            int single_id = omp_get_thread_num();           // id of the thread that will create tasks.

            int single_private_counters[num_thr][k];        // Matrix in which we will save the number of points associated
                                                            // with every centroid (for each thread).

            double single_private_square_dist[num_thr];     // Array in which we will save the sum of all the  square
                                                            // distances between a point and its centroid (for each thread).

            Point single_private_centroids[num_thr][k];     // Matrix in which we will calculate the new centroids
                                                            //  (for each thread).

            double max_dist = 2*pow(MAX_DIST,2);
            double precision = PRECISION;
            double old_mean_square_dist = max_dist;


            // First while: it runs until the stop condition is reached.
            while(my_exit !=1){

                iterations++;
                counter = 0;

                // Second while: the single_id thread starts to create tasks, one for each active thread.
                while (counter < num_thr) {

#pragma omp task default(none) shared(n, num_thr, k, single_private_counters, single_private_centroids,\
                                        single_private_square_dist, iterations) firstprivate(db, centroids, max_dist)

                    {
                        //I've divided the points array in n/num_thr parts, one for each thread.
                        int id = omp_get_thread_num();
                        int dim_m = (int) (n / num_thr);

                        //For each thread I find the start and the stop for the next loop.
                        int start = id * dim_m;
                        int stop = (id + 1) * dim_m;

                        //If the thread is the last one, its stop == n.
                        if (id == num_thr - 1) stop = n;

                        //Temporary variables
                        double d_min, d_temp_x, d_temp_y;
                        int num_centroid;

                        //Local thread variables. They will be used in the next loop.
                        Point thr_private_centr[k];
                        int thr_private_count[k];
                        double thr_private_square_dist = 0.0;


                        // Zero initialization for all the variables we will increase.
                        for (int i = 0; i < k; ++i) {
                            thr_private_count[i] = 0;
                            thr_private_centr[i].x = 0.0;
                            thr_private_centr[i].y = 0.0;
                        }

                        // Here I start to evaluate the cartesian distances.
                        for (int i = start; i < stop; ++i) {
                            d_min = max_dist;
                            //At the end of these second loop I will find the nearest centroid.
                            for (int j = 0; j < k; ++j) {
                                d_temp_x = (db[i].x - centroids[j].x) * (db[i].x - centroids[j].x);
                                d_temp_y = (db[i].y - centroids[j].y) * (db[i].y - centroids[j].y);

                                if (d_temp_x + d_temp_y < d_min) {
                                    d_min = d_temp_x + d_temp_y;
                                    num_centroid = j;                       // num_centroid is the position of the nearest centroid
                                                                            // in centroids array.
                                }
                            }

                            thr_private_square_dist += d_min;               // Increase the sum of the distances.
                            thr_private_count[num_centroid]++;              // Increase the number of point of the cluster.
                            thr_private_centr[num_centroid].x += db[i].x;   // Increase the sum of the x coordinates.
                            thr_private_centr[num_centroid].y += db[i].y;   // Increase the sum of the y coordinates.
                        }


                        // Here I save the private threads results in single_id thread variables.
                        single_private_square_dist[id] = thr_private_square_dist;
                        for (int i = 0; i < k; ++i) {
                            single_private_counters[id][i] = thr_private_count[i];
                            single_private_centroids[id][i].x = thr_private_centr[i].x;
                            single_private_centroids[id][i].y = thr_private_centr[i].y;
                        }
                    }

                    counter++;       //Ready for a new task.
                }

#pragma omp taskwait

                //I will use the counters of single_id thread to gather all the results.
                for (int i = 0; i < k; ++i) {

                    for (int j = 0; j < num_thr; ++j) {
                        if (j == single_id) continue;

                        // For all the other threads I increase the counters.
                        single_private_counters[single_id][i] += single_private_counters[j][i];
                        single_private_centroids[single_id][i].x += single_private_centroids[j][i].x;
                        single_private_centroids[single_id][i].y += single_private_centroids[j][i].y;

                        //For the first iteration, I sum all the square distances.
                        if (i == 0) single_private_square_dist[single_id] += single_private_square_dist[j];
                    }

                    //If a cluster is not empty I calculate the new centroids with the mass centre formula.
                    if (single_private_counters[single_id][i] != 0){
                        centroids[i].x = single_private_centroids[single_id][i].x/single_private_counters[single_id][i];
                        centroids[i].y = single_private_centroids[single_id][i].y/single_private_counters[single_id][i];
                    }

                    //If a cluster is empty, I randomly chose a new centroid.
                    else{
                        centroids[i].x = db[rand()%n+1].x;
                        centroids[i].y = db[rand()%n+1].y;
                    }
                }

                single_private_square_dist[single_id] = single_private_square_dist[single_id]/n;

                // - If you want to evaluate the performances, it is better to use a condition that forces the function
                //   to do the same number of cycles for every call. I've used:
                if (iterations == 40) my_exit = 1;
                // - If you don't want to evaluate the performances, you could use the precision condition:
                //if (fabs(old_mean_square_dist - single_private_square_dist[single_id]) < precision) my_exit = 1;

                old_mean_square_dist = single_private_square_dist[single_id];
            }
        }
    }
}


