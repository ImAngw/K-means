#include "k_means_functions.h"


int graph(Point *db, Point *centroids, Point *true_centroids, int n, int k) {
    /*
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


    // Open a pipe for Gnuplot
    FILE *gnuplotPipe = _popen("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" -persist", "w");

    if (gnuplotPipe == NULL) {
        fprintf(stderr, "I CAN T OPEN Gnuplot.\n");
        return 1;
    }

    // Create point_data.txt
    FILE *dataFile = fopen("point_data.txt", "w");
    if (dataFile == NULL) {
        fprintf(stderr, "I CAN T OPEN point_data.txt.\n");
        return 1;
    }

    // Create centroids_data.txt
    FILE *centroidsFile = fopen("centroids_data.txt", "w");
    if (centroidsFile == NULL) {
        fprintf(stderr, "I CAN T OPEN centroids_data.txt.\n");
        return 1;
    }

    //Create true_centroids_data.txt
    FILE *true_centroidsFile = fopen("true_centroids_data.txt", "w");
    if (true_centroidsFile == NULL) {
        fprintf(stderr, "I CAN T OPEN true_centroids_data.txt.\n");
        return 1;
    }

    //Set the axes scale
    fprintf(gnuplotPipe, "set xrange [-0.1:1.1]\n");
    fprintf(gnuplotPipe, "set yrange [-0.1:1.1]\n");

    //Add all the point in db to point_data.txt
    for (int i = 0; i < n; ++i) fprintf(dataFile, "%lf %lf\n", db[i].x, db[i].y);
    //Add all the point in centroids to centroids_data.txt
    for (int i = 0; i < k; ++i) fprintf(centroidsFile, "%lf %lf\n", centroids[i].x, centroids[i].y);
    //Add all the point in true_centroids to true_centroids_data.txt
    for (int i = 0; i < k; ++i) fprintf(true_centroidsFile, "%lf %lf\n", true_centroids[i].x, true_centroids[i].y);


    fclose(dataFile);
    fclose(centroidsFile);
    fclose(true_centroidsFile);


    // Send instructions with the pipe
    fprintf(gnuplotPipe, "plot 'point_data.txt' using 1:2 with points title 'POINTS',\
                                  'centroids_data.txt' using 1:2  with points pointsize 2 linecolor rgb 'red' title 'CENTROIDS',\
                                    'true_centroids_data.txt' using 1:2 with points lc rgb 'green' title 'TRUE CENTROIDS'\n");

    // Close the pipe
    pclose(gnuplotPipe);

    return 0;
}

double *take_time(Point *db, Point *first_centroids, int n, int k, int n_thr){
    /*
     * db = database the function will use
     * firs_centroids = starting random centroids
     * n = number of points in db
     * k = number of centroids in first_centroids
     * n_thr = max number of threads you want to activate
     *
     * DESCRIPTION:
     * It takes time execution of the function FIND_2D_CENRTORIDS varying at each time the number of active threads,
     * starting from 1 to n_thr,
     */

    Point temp[k];
    double start, stop;
    double *times = (double*) malloc(n_thr*sizeof (double));

    for (int i = 0; i < n_thr; ++i) {

        for (int j = 0; j < k; ++j) {
            temp[j].x = first_centroids[j].x;
            temp[j].y = first_centroids[j].y;
        }

        start = omp_get_wtime();
        find_2D_centroids(db, temp, n, k, i+1);
        stop = omp_get_wtime();

        times[i] = stop - start;
    }
    return times;
}

int speed_plot(double **points, int n_thr, int n_point_in_plot, int *number_of_points){
    /*
     * points[i][j] = matrix of the points where i is the number of thread and j the y coordinate
     * n_thr = max number of active threads
     * n_points_in_plot = number of points (for each thread) the function will represent in the plot
     * number_of points = array with x coordinates
     *
     * DESCRIPTION:
     * It draws the 2D graph of the speed execution for each thread.
     * You need gnuplot installed in "C:\Program Files\gnuplot" to run this.
     */

    FILE *gnuplotPipe = _popen("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" -persist", "w");

    if (gnuplotPipe == NULL) {
        fprintf(stderr, "I CAN T OPEN Gnuplot.\n");
        return 1;
    }

    FILE *dataFile = fopen("point_data.txt", "w");
    if (dataFile == NULL) {
        fprintf(stderr, "I CAN T OPEN point_data.txt.\n");
        return 1;
    }


    int max_x_value;
    int min_x_value;

    double max_y_value = points[0][0];
    double min_y_value = points[0][0];


    if (n_point_in_plot != 1) {
        min_x_value = number_of_points[0];
        max_x_value = number_of_points[n_point_in_plot-1];
    }
    else {
        min_x_value = 0;
        max_x_value = 2*number_of_points[n_point_in_plot-1];
    }


    // I find max values to scale the cartesian axes
    for (int i = 0; i < n_point_in_plot; ++i) {
        if (max_x_value < number_of_points[i]) max_x_value = number_of_points[i];
        if (min_x_value > number_of_points[i]) min_x_value = number_of_points[i];
        for (int j = 0; j < n_thr; ++j) {
            if (max_y_value < points[j][i]) max_y_value = points[j][i];
            if (min_y_value > points[j][i]) min_y_value = points[j][i];
        }
    }


    // I write the points on dataFile
    for (int i = 0; i < n_point_in_plot; ++i) {
        for (int set = 0; set < n_thr; ++set) fprintf(dataFile, "%d %lf\t", number_of_points[i], points[set][i]);
        fprintf(dataFile, "\n");
    }

    fclose(dataFile);

    // I set the scale and name of cartesian axes
    fprintf(gnuplotPipe, "set yrange [0:%lf]\n", max_y_value + (double)max_y_value/10);
    fprintf(gnuplotPipe, "set xrange [%d:%d]\n", min_x_value - (int)((max_x_value-min_x_value)/10),\
                                                                max_x_value + (int)((max_x_value-min_x_value)/10));

    fprintf(gnuplotPipe, "set xlabel 'Number of points'\n");
    fprintf(gnuplotPipe, "set ylabel 'Time [s]'\n");



    // Here I start to plot the graph
    fprintf(gnuplotPipe, "plot");
    for (int i = 0; i < n_thr; i++) {
        fprintf(gnuplotPipe, " 'point_data.txt' using %d:%d with points title '%d THR' ", 2*i+1, 2*i+2, i+1);
        fprintf(gnuplotPipe, ",");
        fprintf(gnuplotPipe, " '' using %d:%d with lines title ''", 2*i+1, 2*i+2);

        if (i < n_thr - 1) {
            fprintf(gnuplotPipe, ",");
        }
    }

    fprintf(gnuplotPipe, "\n");

    // Close the pipe
    pclose(gnuplotPipe);

    return EXIT_SUCCESS;

}

int speedup_plot(double **points, int n_thr, int n_point_in_plot){
    /*
    * points[i][j] = matrix of the points where i is the number of thread and j the y coordinate
    * n_thr = max number of active threads
    * n_points_in_plot = number of points (for each thread) the function will represent in the plot
    *
     * DESCRIPTION:
    * It draws the 2D graph of the speedup for each thread.
    * You need gnuplot installed in "C:\Program Files\gnuplot\bin\" to run this.
    */


    FILE *gnuplotPipe = _popen("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" -persist", "w");

    if (gnuplotPipe == NULL) {
        fprintf(stderr, "I CAN T OPEN Gnuplot.\n");
        return 1;
    }

    FILE *dataFile = fopen("speedup_data.txt", "w");
    if (dataFile == NULL) {
        fprintf(stderr, "I CAN T OPEN speedup_data.txt.\n");
        return 1;
    }

    // Values I use to scale ht cartesian axes
    int max_x_value = n_thr+2;
    int max_y_value = 10;


    // I write data on dataFile
    for (int i = 0; i < n_thr; ++i) {
        fprintf(dataFile, "%d %lf\n", i+1, points[0][n_point_in_plot-1]/points[i][n_point_in_plot-1]);
    }

    fclose(dataFile);

    // I set the scale and name of cartesian axes
    fprintf(gnuplotPipe, "set yrange [0:%d]\n", max_y_value);
    fprintf(gnuplotPipe, "set xrange [0:%d]\n", max_x_value);

    fprintf(gnuplotPipe, "set xlabel 'Threads'\n");
    fprintf(gnuplotPipe, "set ylabel 'Speed up'\n");


    // Here I start to plot the graph
    fprintf(gnuplotPipe, "plot 'speedup_data.txt' using 1:2 with points title ''");
    fprintf(gnuplotPipe, ",");
    fprintf(gnuplotPipe, " '' using 1:2 with lines title ''\n");

    // Close the pipe
    pclose(gnuplotPipe);

    return EXIT_SUCCESS;

}

int speedup_curves(int *number_of_points, int k, int n_thr, int n_point_in_plot){
    /*
     * number_of points = array with x coordinates
     * k = number of centroids
     * n_thr = max number of active threads
     * n_points_in_plot = number of points (for each thread) the function will represent in the plot
     *
     * DESCRIPTION:
     * It draws the 2D graph of the speedup and the 2D graph of speed execution for each thread.
     * You need gnuplot installed in "C:\Program Files\gnuplot\bin\" to run this.
     */

    srand(time(NULL));
    Point *true_centroids = centroids_initialize(k);
    Point *database;
    Point *centroids = centroids_initialize(k);

    double *times;

    double **plot_points = (double **) malloc(n_thr*sizeof (double*));
    if (plot_points == NULL) return 1;

    for (int i = 0; i < n_thr; ++i) {
        plot_points[i] = (double*) malloc(n_point_in_plot*sizeof (double));
        if (plot_points[i] == NULL) return 1;
    }

    // for each point I create a new database and I take time of all the executions
    for (int i = 0; i < n_point_in_plot; ++i) {
        database = new_2D_database(number_of_points[i], k, true_centroids);
        times = take_time(database, centroids, number_of_points[i], k, n_thr);

        for (int j = 0; j < n_thr; ++j) plot_points[j][i] = times[j];

    }


    speed_plot(plot_points, n_thr, n_point_in_plot, number_of_points);
    speedup_plot(plot_points, n_thr, n_point_in_plot);


    free(true_centroids);
    free(database);
    free(centroids);
    free(times);

    for (int i = 0; i < n_thr; ++i) free(plot_points[i]);
    free(plot_points);

    return 0;
}

