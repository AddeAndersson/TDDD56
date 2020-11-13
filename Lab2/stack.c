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

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 

  // Test TODO: We were here
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
stack_push(int* task)
{
  struct node* new_node = (struct node*)malloc(sizeof(struct node));
  
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  new_node->task = *task;
  pthread_mutex_lock(&mutex);
  new_node->prev = stack->current_node;
  stack->current_node = new_node;
  pthread_mutex_unlock(&mutex);


#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
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
stack_pop()
{

  if(stack->current_node == NULL) return 1;

#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&mutex);
  node_t* old_node = stack->current_node;
  stack->current_node = old_node->prev;
  pthread_mutex_unlock(&mutex);
  free(old_node);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

