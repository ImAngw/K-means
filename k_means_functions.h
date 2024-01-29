#ifndef K_MEANS_K_MEANS_FUNCTIONS_H
#define K_MEANS_K_MEANS_FUNCTIONS_H

#include "omp.h"
#include "time.h"
#include "stdlib.h"
#include <stdio.h>
#include "math.h"


typedef struct {
    double x;
    double y;
} Point;


Point* centroids_initialize(int k);
/* VARIABLES:
 * k = number of centroids you want to initialize
 *
 * DESCRIPTION:
 * It initializes a struct Point array of k dimension with normalized random point (in [0;1] interval)
*/

Point* new_2D_database( int n, int k, Point *centroids);
/* VARIABLES:
 * n = number of point in the database
 * k = number of centroids you want to use
 * centroids = struct Point array with centroids you want to use
 *
 * DESCRIPTION:
 * It takes the true centroids you want to use for your database and for each of them
 * it produces n/k point with deviation DEV (x_1 +/- DEV; y_1 +/- DEV).
 */

void find_2D_centroids(Point* db, Point* centroids, int n, int k, int num_thr);
/* VARIABLES:
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

int graph(Point *db, Point *centroids, Point *true_centroids, int n, int k);
/* VARIABLES:
 * db = database (strut Point array of dimension n) you want to use
 * centroids = struct Point array of dimension k, with all the founded centroids
 * true_centroids = struct Point array of dimension k, with all the true centroids used to initialize the database db
 * n = number of points in database
 * k = number of points in centroids
 *
 * DESCRIPTION:
 * It draws the 2D graph of the point and centroids.
 * You need gnuplot installed in "C:\Program Files\gnuplot\bin\" to run this.
 */

double *take_time(Point *db, Point *first_centroids, int n, int k, int n_thr);
/* VARIABLES:
 * db = database the function will use
 * firs_centroids = starting random centroids
 * n = number of points in db
 * k = number of centroids in first_centroids
 * n_thr = max number of threads you want to activate
 *
 * DESCRIPTION:
 * It takes time execution of the function find_2D_centroids varying at each time the number of active threads,
 * starting from 1 to n_thr,
 */

int speed_plot(double **points, int n_thr, int n_point_in_plot, int *number_of_points);
/* VARIABLES:
 * points[i][j] = matrix of the points where i is the number of thread and j the y coordinate
 * n_thr = max number of active threads
 * n_points_in_plot = number of points (for each thread) the function will represent in the plot
 * number_of points = array with x coordinates
 *
 * DESCRIPTION:
 * It draws the 2D graph of the speed execution for each thread.
 * You need gnuplot installed in "C:\Program Files\gnuplot" to run this.
 */

int speedup_plot(double **points, int n_thr, int n_point_in_plot);
/* VARIABLES:
 * points[i][j] = matrix of the points where i is the number of thread and j the y coordinate
 * n_thr = max number of active threads
 * n_points_in_plot = number of points (for each thread) the function will represent in the plot
 *
 * DESCRIPTION:
 * It draws the 2D graph of the speedup for each thread.
 * You need gnuplot installed in "C:\Program Files\gnuplot\bin\" to run this.
*/

int speedup_curves(int *number_of_points, int k, int n_thr, int n_point_in_plot);
/* VARIABLES:
 * number_of points = array with x coordinates
 * k = number of centroids
 * n_thr = max number of active threads
 * n_points_in_plot = number of points (for each thread) the function will represent in the plot
 *
 * DESCRIPTION:
 * It draws the 2D graph of the speedup and the 2D graph of speed execution for each thread.
 * You need gnuplot installed in "C:\Program Files\gnuplot\bin\" to run this.
 */

#endif //K_MEANS_K_MEANS_FUNCTIONS_H
