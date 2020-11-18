/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif


// Create and allocate pool
// static node_t* pool[NB_THREADS];
// _Atomic static size_t cur;

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 

  // Test
  node_t* prev_ptr = stack->current_node;
  int counter = 0;
  while(prev_ptr != NULL) {
    prev_ptr = prev_ptr->prev;
    counter++;
  }

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(node_t* n)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&mutex);
  n->prev = stack->current_node;
  stack->current_node = n;
  pthread_mutex_unlock(&mutex);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  node_t* current_node;
  do {
    current_node = stack->current_node;
    n->prev = current_node;
  } while((size_t)current_node != cas((size_t*)&stack->current_node, (size_t)current_node, (size_t)n));
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check(stack);

  return 0;
}

int /* Return the type you prefer */
stack_pop(node_t** n)
{
  node_t* top = stack->current_node;
  
  if(top == NULL) return 1;

#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&mutex);
  *n = stack->current_node;
  stack->current_node = top->prev;
  pthread_mutex_unlock(&mutex);
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  node_t* current_node;
  node_t* prev;
  do {
    current_node = stack->current_node;
    prev = current_node->prev;
  } while((size_t)current_node != cas((size_t*)&stack->current_node, (size_t)current_node, (size_t)prev));

  *n = current_node;
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

void
stack_pool_init()
{
  stack = malloc(sizeof(stack_t));
  for(size_t i = 0; i < NB_THREADS; ++i) {
    for(size_t j = 0; j < MAX_PUSH_POP; ++j) {
      node_t *n = (node_t*)malloc(sizeof(node_t));
      n->prev =  pool[i];
      n->task = j;
      pool[i] = n;
    }
    //pool[i] = malloc(sizeof(node_t) * (MAX_PUSH_POP / NB_THREADS));
  } 
}

void
stack_pool_free() {
  // Free stack
  while(stack->current_node != NULL) {
    node_t* n = stack->current_node;
    stack->current_node = n->prev;
    free(n);
  }

  free(stack);

  // Free pool
  for(int i = 0; i < NB_THREADS; ++i) {
    while(pool[i] != NULL) {
      node_t* n = pool[i];
      pool[i] = pool[i]->prev;
      free(n);
    }
  }
}

void
stack_fill(size_t size) {
  for(int i = 0; i < size; ++i) {
      node_t *n = (node_t*)malloc(sizeof(node_t));
      n->task = i;
      stack_push(n);
  }
}