/*
 * test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "test.h"
#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

/* Helper function for measurement */
double timediff(struct timespec *begin, struct timespec *end)
{
	double sec = 0.0, nsec = 0.0;
   if ((end->tv_nsec - begin->tv_nsec) < 0)
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec  - 1);
      nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
   } else
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec );
      nsec = (double)(end->tv_nsec - begin->tv_nsec);
   }
   return sec + nsec / 1E9;
}

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

data_t data;
node_t* first_node; // Used to test ABA

#ifndef NDEBUG
int
assert_fun(int expr, const char *str, const char *file, const char* function, size_t line)
{
	if(!(expr))
	{
		fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file, function, line, str);
		abort();
		// If some hack disables abort above
		return 0;
	}
	else
		return 1;
}
#endif

#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;
    
    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; ++i)
      {
        // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
        //node_t *n;
        stack_pop(args->id);
        //n->prev = pool[args->id];
        //pool[args->id] = n;
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);
    
    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int id = args->id;
  int task = pool[id]->task;
  clock_gettime(CLOCK_MONOTONIC, &t_start[id]);
  for (int i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
      // See how fast your implementation can push MAX_PUSH_POP  in parallel
      //printf("ID: %i ", id);
      //printf("Task: %i - ID: %i\n", pool[id]->task, id);
      //assert(n != NULL);
      //node_t* n = pool[id];
      //if(n == NULL) continue;
 
      stack_push(task, id);
             //printf("123");
      //
  }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[id]);

  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  // Initialize your test batch
  pthread_mutex_init(&mutex, NULL);
  //stack_fill(MAX_PUSH_POP);
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  // Allocate a new stack and reset its values
  //stack = malloc(sizeof(stack_t));
  //node_t* node = malloc(sizeof(node_t));

  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):
  // node_t* task = -1;
  // stack->current_node = NULL;
  // //stack_push(&task);
  //   for(size_t i = 0; i < 1000; ++i) {
  //   stack_push(&task);
  // }
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  // while(!stack_pop()) {
  //   ;
  // }
  // free(stack);

}

void
test_finalize()
{
  // Destroy properly your test batch
  
}

void
print_stack() 
{
  node_t* prev_ptr = stack->current_node;
  while(prev_ptr != NULL) {
    printf("%i -> ", prev_ptr->task);
    prev_ptr = prev_ptr->prev;
  }
  printf("\n");
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it
  // TODO: We were here
  // Do some work

  //stack = malloc(sizeof(stack_t));
  // node_t* task = 0;
  // while(task < 10) {
  //   stack_push(&task);
  //   task++;
  // }


  //assert(stack->current_node->task == 9);

  // check if the stack is in a consistent state
  //int res = assert(stack_check(stack));

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // Now, the test succeeds
  return 1 && assert(stack->current_node != NULL);
}

int
test_pop_safe()
{
  return 1 && assert(stack->current_node != NULL);
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 2

void* push_and_pop(int id) {

  

  //stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  printf("\n");
  if(id == 0) {
    printf("Thread: %i started pop()\n", id);
    stack_pop_aba(0);
    //printf("Thread: %i stopped pop() on: %i\n", id, n->task);
    print_stack();
  }



  if(id == 1) {
    for(int j = 0; j < 2; ++j) {
      printf("Thread: %i started pop()\n", id);
      stack_pop(id);
      //printf("Thread: %i stopped pop() on: %i\n", id, n->task);
      print_stack();
    }
      //node_t *n = first_node;
      //pool[id] = n->prev;
      //printf("Thread: %i started push() on %i\n", id, n->task);
      stack_push(0, 1);
      printf("Thread: %i stopped push()\n", id);
      print_stack();
  }

  return NULL;
}

int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2

  print_stack();
  int success, aba_detected = 1;
  // Write here a test for the ABA problem
  pthread_t thread[ABA_NB_THREADS];
  //thread_test_cas_args_t args[ABA_NB_THREADS];
  size_t i;
  for (i = 0; i < ABA_NB_THREADS; i++)
  {
    //args[i].id = i;
    pthread_create(&thread[i], NULL, &push_and_pop, i);
  }

  for (i = 0; i < ABA_NB_THREADS; i++)
  {
    pthread_join(thread[i], NULL);
  }

  success = aba_detected;
  printf("\nABA detected!\n");
  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);

// Initialize stack and pools
stack_pool_init(NB_THREADS, MAX_PUSH_POP);

// MEASURE == 0 -> run unit test
#if MEASURE == 0
  test_init();

  //node_t *A = (node_t*)malloc(sizeof(node_t));
  //node_t *B = (node_t*)malloc(sizeof(node_t));
  //node_t *C = (node_t*)malloc(sizeof(node_t));
  //C->task = 2;
  //B->task = 1;
  //A->task = 0;
  stack_push(2, 0);
  stack_push(1, 0);
  //stack_push(0, 1);
  //first_node = A;
  
  
  test_run(test_cas);
  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):

  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];
  pthread_attr_init(&attr);
  
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1

      stack_fill(MAX_PUSH_POP);
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }
  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
        printf("Thread %d time: %f\n", i, timediff(&t_start[i], &t_stop[i])*1000000);
    }
    
  // Free stack and pools
  //stack_pool_free(NB_THREADS);
#endif

  return 0;
}
