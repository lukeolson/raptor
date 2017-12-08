#include <stdio.h> 
#include <sys/types.h>
#include <sys/times.h>
#include <time.h>
#include <sys/time.h>

#ifndef CLK_TCK
#define CLK_TCK 100
#endif

#ifndef CLOCKS_PER_SEC 
#define CLOCKS_PER_SEC 1000000
#endif 

double sys_timer_CLOCK() {
  clock_t tmp;
  tmp = clock();
  return (double) tmp/(CLOCKS_PER_SEC);
}

double sys_timer() {
    struct tms use;
    clock_t tmp;
    times(&use);
    tmp = use.tms_utime + use.tms_stime;
    return (double)(tmp) / CLK_TCK;
}

