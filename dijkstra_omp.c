/* assert */
#include <assert.h>
/* INFINITY */
#include <math.h>
/* FILE, fopen, fclose, fscanf, rewind */
#include <stdio.h>
/* EXIT_SUCCESS, malloc, calloc, free */
#include <stdlib.h>
/* time, CLOCKS_PER_SEC */
#include <time.h>
#include <omp.h>

#define ROWMJR(R,C,NR,NC) (R*NC+C)
#define COLMJR(R,C,NR,NC) (C*NR+R)
/* define access directions for matrices */
#define a(R,C) a[ROWMJR(R,C,ln,n)]
#define b(R,C) b[ROWMJR(R,C,nn,n)]

int num_threads;

static void load(
  char const * const filename,
  int * const np,
  float ** const ap
)
{
  int i, j, n, ret;
  FILE * fp=NULL;
  float * a;

  /* open the file */
  fp = fopen(filename, "r");
  assert(fp);

  /* get the number of nodes in the graph */
  ret = fscanf(fp, "%d", &n);
  assert(1 == ret);

  /* allocate memory for local values */
  a = malloc(n*n*sizeof(*a));
  assert(a);

  /* read in roots local values */
  for (i=0; i<n; ++i) {
    for (j=0; j<n; ++j) {
      ret = fscanf(fp, "%f", &a(i,j));
      assert(1 == ret);
    }
  }

  /* close file */
  ret = fclose(fp);
  assert(!ret);

  /* record output values */
  *np = n;
  *ap = a;
}


static void dijkstra(
  int const s,
  int const n,
  float const * const a,
  float ** const lp
)//(source, total nodes count, graph, output array)
{
  int nth, chunk, mv, k;
  float md;
  int * m;
  float * l;

  m = calloc(n, sizeof(*m)); // visited nodes
  assert(m);

  l = malloc(n*sizeof(*l));
  assert(l);

  for (k=0; k<n; ++k) {
    l[k] = a(k,s); //init path from source; including direct path and inf
  }

  #pragma omp parallel shared(mv, md)
  {
    int i, j;
    struct float_int {
      float l;
      int u;
    } min;
    int startv,endv, me = omp_get_thread_num();  // my thread number

    #pragma omp single
    {  
      nth = omp_get_num_threads();  
      chunk = n/nth;  
      printf("there are %d threads\n",nth);  
    }

    startv = me * chunk;
    endv = startv + chunk - 1;
    #pragma omp single
    {
    m[s] = 1; // visited source nodes
    }

    min.u = -1; /* avoid compiler warning */

    for (i=0; i<n; ++i) {
      #pragma omp single 
      {  md = INFINITY; mv = 0;  }

      /* find local minimum */
      min.l = INFINITY;
      for (j=startv; j<=endv; ++j) {
        if (!m[j] && l[j] < min.l) { //the node is not visited and its min d so far is the smallest
          min.l = l[j];
          min.u = j;
        }
      }

      #pragma omp critical
      {  if (min.l < md)  
        {  md = min.l; mv = min.u;  }
      }
      #pragma omp barrier

      // mark new vertex as done 
      #pragma omp single 
      {  m[mv] = 1;  }

      for (j=startv; j<=endv; j++) {
        if (!m[j] && md+a(j,mv) < l[j])
          l[j] = md+a(j,mv);
      }
      #pragma omp barrier 
    }
  }

  free(m);

  *lp = l;
}

static void print_time(double const seconds)
{
  printf("Search Time: %0.06fs\n", seconds);
}

static void print_numbers(
  char const * const filename,
  int const n,
  float const * const numbers)
{
  int i;
  FILE * fout;

  /* open file */
  if(NULL == (fout = fopen(filename, "w"))) {
    fprintf(stderr, "error opening '%s'\n", filename);
    abort();
  }

  /* write numbers to fout */
  for(i=0; i<n; ++i) {
    fprintf(fout, "%10.4f\n", numbers[i]);
  }

  fclose(fout);
}

int main(int argc, char ** argv)
{
  int n;
  clock_t ts, te;
  float * a, * l;

  if(argc < 5){
     printf("Invalid number of arguments.\nUsage: dijkstra <graph> <source> <output_file> <num_threads>.\n");
     return EXIT_FAILURE;
  }

  num_threads = atoi(argv[4]);
  load(argv[1], &n, &a); //(input name, total nodes count, graph)
  // printf("%d, %d, %d\n", n, omp_get_num_threads(), (n % omp_get_num_threads()));
  if((n % num_threads) > 0){
     printf("Invalid number of threads.\n");
     return EXIT_FAILURE;
  }
  omp_set_num_threads(num_threads);

  ts = clock();
  dijkstra(atoi(argv[2]), n, a, &l); //(source, total nodes count, graph, output array)
  te = clock();

  print_time((double)(te-ts)/CLOCKS_PER_SEC);
  print_numbers(argv[3], n, l);

  free(a);
  free(l);

  return EXIT_SUCCESS;
}
