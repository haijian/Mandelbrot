/*
  This program draws the Mandelbrot Set. 
  It is based on the master-slave structure.

  The master first sends a row number to each slave. It also 
  keep the count of how many lines have been sent and how many
  lines have been received from the slaves. Immediately after
  It receives the calculated mandelbrot numbers of a line, it 
  draws that line and sends next line the ready slave from whom
  it received the message.
  When the user drag an area to zoom in, it broadcasts the new lower
  left and upper right points to all the slaves. if the user drag a
  quite small area(less than 10*10 pixes), it is time to terminate.
  The master sends impossible lower left and upper right points to slaves
  to indicate the termination.

  The slave receives the line number from the master, then calcualtes the
  mandelbrot numbers for all the points in that line and then sends the
  result back the master.
  If the line number is negtive, it indicates that a window is ready, the 
  slave should wait for the new lower left and upper right points.
  if the new lower left and upper right points are (-3, -3), (-3, -3), then
  it is time to terminate. else the slave continue to calculate the 
  mandelbrot numbers.

  The master uses MPI_Send to send the line numbers to a slave, uses 
  MPI_Recv to receive calculated mandelbrot numbers from a slave.
  The Slave uses MPI_Send to send the calculated mandelbrot numbers
  to master and uses the MPI_Recv the receive line number from master.

  The master and slaves use MPI_Bcast to send and receive the new 
  lower left and upper right points.
  
  
  Compile with 'mpicc -mpe=graphics Mandelbrot.c -o Mandelbrot -lm'

 */
#include <mpi.h>
#define MPE_GRAPHICS
#include <mpe.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_COUNT 255
#define W 800
#define H 800

static MPE_XGraph graph;
static MPE_Color color[256];
static int tag=1;
MPI_Status status;
double min_max[4];/*used to for the master process to broadcast the new lower left and upper right values to slaves.
		    0:real_min, 1:real_max, 2: imag_min, 3:imag_max */
double real_min, real_max, imag_min, imag_max, real_scale, imag_scale;
int* mans;/* used for a slave process to send to the mandelbrot number of the points of a line. 
	     the last number is used to indicate the line number;*/

int np, id; //number of processes and current process id;

/*structure for the complex type*/
typedef struct{
  double real;
  double imag;
}Complex_type;

/*initiate the Environment*/
void initEnv(int argc, char** argv){
  int err;

  err = MPI_Init(&argc, &argv); /* Initialize MPI */
  if (err != MPI_SUCCESS) {
    printf("MPI_init failed!\n");
    exit(1);
  }

  err = MPI_Comm_size(MPI_COMM_WORLD, &np);	/* Get nr of tasks */
  if (err != MPI_SUCCESS) {
    printf("MPI_Comm_size failed!\n");
    MPI_Finalize();
    exit(1);
  }
  
  if(np < 2){
    printf("At least 2 processes are needed\n");
    MPI_Finalize();
    exit(1);
  }

  err = MPI_Comm_rank(MPI_COMM_WORLD, &id);    /* Get id of this process */

  if (err != MPI_SUCCESS) {
    printf("MPI_Comm_rank failed!\n");
    MPI_Finalize();
    exit(1);
  }
  
  MPE_Open_graphics(&graph, MPI_COMM_WORLD, NULL, 0, 0, W, H, 0 );
  
  MPE_Make_color_array(graph, 256, color);
}

/* to calculate the mandelbrot number of a complex */
int calc_mandel(Complex_type c){
  int count = 0;
  Complex_type z;
  double len2, temp;
  z.real = z.imag = 0.0;
  do{
    temp = z.real*z.real - z.imag*z.imag + c.real;
    z.imag = 2.0*z.real*z.imag  + c.imag;
    z.real = temp;
    len2 = z.real*z.real + z.imag*z.imag;
    if(len2  > 4.0) break;
    count++;
  }while(count < MAX_COUNT);
  return count;
}

/*used for the master process to draw a line.
  mans is the array sent by a slave process.
  the array contains the mandelbrot number of
  points of the line and the line number.*/
void draw_line(int* mans, MPE_Point* points){
  int lnum = mans[W];
  int i;
  for(i = 0; i < W; i++){
    points[i].x = i;
    points[i].y = lnum;
    points[i].c = color[mans[i]];
  }
  MPE_Draw_points(graph, points, W);
  MPE_Update(graph);
}

//master process
void master(int np){
  int window = 0;//which window;
  int l_sent = 0;//how many lines have sent;
  int l_recv = 0;//how many lines have received;
  int i;
  //the points of a line.
  MPE_Point* points = (MPE_Point*)malloc(W*sizeof(MPE_Point));
  double start, stop;
  do{
    start = MPI_Wtime();
    window++;

    //first send a line number to each slave
    for(i = 1; i < np; i++){
      if(l_sent < H){
	MPI_Send(&l_sent, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
	l_sent++;
      }
    };

    do{
      /*Receive the result back from the slaves, we use MPI_ANY_SOURCE here
	because we just receive the result from the fastest slave*/
      MPI_Recv(mans, W+1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);

      /*after we received a computed mandelbrot number of points of a line, 
	the master process begin to draw that line and increase the received
	lines variable*/
      draw_line(mans,points);
      l_recv++;
      
      /*if there are still some rows to send, send the next row to the free slave*/
      if(l_sent < H){
	MPI_Send(&l_sent, 1, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
	l_sent++;
      }else{
	/*if we have sent all the rows, then the end signal to the free slave 
	  to tell that it can have a rest now.
	  we use -1 which is an impossible line number to represent the end signal
	*/
	int endSignal = -1;
	MPI_Send(&endSignal, 1, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
      }
    }while(l_recv < H);/*if we have received results back of all the rows from slaves,
			 then current window is ready, we can wait for the user to drag
			 an area to zoom in*/
    stop = MPI_Wtime();
    printf("Window %d takes %f seconds\n", window, stop-start);//how long it takes to draw a window

    int px, py, rx, ry;
      
    /*get the drag region from the user.
      It is interesting that I found (px,py) is always the lower left poit
      and (rx, ry) is always the upper right point no matter I drag from left
      to right or from right to left*/
    MPE_Get_drag_region(graph, MPE_BUTTON1, MPE_DRAG_SQUARE, &px, &py, &rx, &ry);
    /*
      When the user indicates that it is time to terminate, the master
      sends a stop-signal to all slaves.
      User indicates termination by selecting a very small area(less than 10*10 pixels) to zoom in to,
      e.g. clicks with cursor in the window instead of dragging a square area.
     */
    if((rx-px)*(ry-py) < 10*10){
      /*
	when it is the time to terminate the program, we broadcast the impossible lower left(-3, -3)
	and upper right point(-3, -3) to indicate the slaves that it is time to terminate.
       */
      min_max[0]=min_max[1]=min_max[2]=min_max[3]=-3.0;
      MPI_Bcast(min_max, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
      break;
    }else{
      /*
	when user select a area to zoom in, the master recalculate the lower left and upper right points
	and broadcast these two points to the slaves.
       */
      min_max[0] = real_min + px*real_scale;
      min_max[1] = real_min + rx*real_scale;
      min_max[2] = imag_min + py*imag_scale;
      min_max[3] = imag_min + ry*imag_scale;
      real_min = min_max[0];
      real_max = min_max[1];
      imag_min = min_max[2];
      imag_max = min_max[3];
      real_scale = (real_max - real_min) / (double)W;
      imag_scale = (imag_max - imag_min) / (double)H;

      /* reset the send counter and received counter*/
      l_sent = 0;
      l_recv = 0;
      MPI_Bcast(min_max, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  }while(1);
  free(points);
}

void slave(){
  int rn, i;
  int imag, real;
  Complex_type cx;
  do{
    do{
      /*
	receive the line number from the master process, if the line number is
	negtive, it indicates that a window is ready, it should wait*/
      MPI_Recv(&rn, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
      if(rn == -1) break;
      
      /*calculate the mandelbrot numbers of points of the line.*/
      cx.imag = imag_min + ((double)rn*imag_scale);
      for(i = 0; i < W; i++){
	cx.real = real_min + ((double)i*real_scale);
	mans[i] = calc_mandel(cx);
      }
      mans[W] = rn;
      
      /*send the result back to the master process*/
      MPI_Send(mans, W+1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }while(1);
    
    /*after a window is ready, the slave waits for the new 
      lower left and upper right points, if the points are
      (-3, -3),(-3, -3), the slave terminates else it continues
      to calculate the mandelbrot numbers for new lines.
    */
    MPI_Bcast(min_max, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    real_min = min_max[0];
    real_max = min_max[1];
    imag_min = min_max[2];
    imag_max = min_max[3];
    imag_scale = (imag_max-imag_min)/(double)H;
    real_scale = (real_max-real_min)/(double)W;
  }while(real_min!=-3&&real_max!=-3&&imag_min!=-3&&imag_max!=-3);
}
int main(int argc, char** argv){
  real_min=imag_min=-2.0;
  real_max=imag_max=2.0;
  real_scale = (real_max-real_min)/(double)W;
  imag_scale = (imag_max-imag_min)/(double)H;
  mans = (int*)malloc((W+1)*sizeof(int));/* to store the mandelbrot number of the points of a line. the last number 
					    is used to indicate the line number;*/

  initEnv(argc, argv); //init the environment;

  id ? slave() : master(np); //if id is 0, then execute the master process, else the slave process;
  
  /* Close the graphics window */
  MPE_Close_graphics (&graph);

  /* Free storage */
  free(mans);

  /* Exit */
  MPI_Finalize();
  exit(0);
}
